#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import textwrap
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError, conint

from call_llm import query_llm


# ============================================================
# ---------------- Pydantic Output Schema --------------------
# ============================================================

class ScoredEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity: StrictStr
    significance_score: conint(ge=0, le=4)
    rationale: StrictStr


class ClinicalSignificanceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scored_entities: List[ScoredEntity]


class PromptYAML(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: StrictStr


_CLINICAL_SIGNIFICANCE_SCHEMA = ClinicalSignificanceOutput.model_json_schema()
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


# ============================================================
# ---------------- JSON Helpers ------------------------------
# ============================================================

def sanitize_json_text(s: str) -> str:
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return _TRAILING_COMMA_RE.sub(r"\1", s).strip()


def extract_first_json_value(text: str) -> Optional[str]:
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return None

    start = obj_start if (obj_start != -1 and (arr_start == -1 or obj_start < arr_start)) else arr_start
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"

    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def parse_json_from_raw(raw: str) -> Any:
    json_text = sanitize_json_text(extract_first_json_value(raw) or raw)
    return json.loads(json_text)


# ============================================================
# ---------------- Validation --------------------------------
# ============================================================

def _normalize_to_object(parsed: Any) -> Dict[str, Any]:
    if isinstance(parsed, list):
        return {"scored_entities": parsed}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Parsed JSON must be an object or array, got {type(parsed)}")


def _validate_and_check_entities(
    *,
    entities: List[Dict[str, Any]],
    parsed: Any,
) -> ClinicalSignificanceOutput:
    obj = _normalize_to_object(parsed)
    validated = ClinicalSignificanceOutput.model_validate(obj)

    input_entities = {e["entity"] for e in entities}
    output_entities = {e.entity for e in validated.scored_entities}

    if input_entities != output_entities:
        raise ValueError(
            "Mismatch between input and output entities.\n"
            f"Missing: {sorted(input_entities - output_entities)}\n"
            f"Extra: {sorted(output_entities - input_entities)}"
        )

    seen: set = set()
    dedup: List[ScoredEntity] = []
    for se in validated.scored_entities:
        if se.entity not in seen:
            seen.add(se.entity)
            dedup.append(se)
    return ClinicalSignificanceOutput(scored_entities=dedup)


# ============================================================
# ---------------- Prompt Building ---------------------------
# ============================================================

def _load_prompt_yaml(path: str) -> PromptYAML:
    return PromptYAML.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def build_messages(system_prompt: str, ground_truth_report: str, entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    user_prompt = (
        f"Ground Truth Report:\n{ground_truth_report}\n\n"
        f"Entities JSON:\n{json.dumps(entities, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_significance_repair_messages(
    *,
    required_entities: List[str],
    error_msg: str,
    raw_response: str,
) -> List[Dict[str, str]]:
    repair_system = (
        "You are a strict JSON repair assistant.\n"
        "Return ONLY valid JSON (no markdown, no commentary).\n"
        "Output MUST be either a JSON object with key 'scored_entities' or a JSON array.\n"
        "Each item must match the JSON schema provided.\n"
        "You must include EXACTLY the required entities (no missing, no extras).\n"
        "If duplicates exist, keep only the first occurrence.\n"
    )
    repair_user = (
        "Fix the model output to satisfy all constraints.\n\n"
        "Required entities (exactly once each):\n"
        f"{json.dumps(required_entities, indent=2)}\n\n"
        "JSON Schema:\n"
        f"{json.dumps(_CLINICAL_SIGNIFICANCE_SCHEMA, indent=2)}\n\n"
        "Validation error:\n"
        f"{error_msg}\n\n"
        "Original model output:\n"
        f"{raw_response}\n"
    )
    return [
        {"role": "system", "content": repair_system},
        {"role": "user", "content": repair_user},
    ]


# ============================================================
# ---------------- Core Workflow -----------------------------
# ============================================================

def validate_significance_with_one_repair(
    *,
    entities: List[Dict[str, Any]],
    raw_output: str,
    model_name: Optional[str],
    token_path: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    required_entities = [e["entity"] for e in entities]

    try:
        validated = _validate_and_check_entities(entities=entities, parsed=parse_json_from_raw(raw_output))
        return validated.model_dump()
    except (ValueError, ValidationError, JSONDecodeError, TypeError) as e:
        repair_messages = build_significance_repair_messages(
            required_entities=required_entities,
            error_msg=str(e),
            raw_response=raw_output,
        )
        repaired_raw = query_llm(
            messages=repair_messages,
            model=model_name,
            token_path=token_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            validated2 = _validate_and_check_entities(entities=entities, parsed=parse_json_from_raw(repaired_raw))
            return validated2.model_dump()
        except (ValueError, ValidationError, JSONDecodeError, TypeError) as repair_exc:
            raise RuntimeError(f"Repair attempt also failed: {repair_exc!r}") from repair_exc


def run_clinical_significance_workflow(
    ground_truth_report: str,
    entities: List[Dict[str, Any]],
    prompt_yaml_path: str,
    model_name: Optional[str] = None,
    token_path: str = "",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    system_prompt = _load_prompt_yaml(prompt_yaml_path).prompt
    messages = build_messages(system_prompt, ground_truth_report, entities)

    raw_output = query_llm(
        messages=messages,
        model=model_name,
        token_path=token_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if not isinstance(raw_output, str):
        raise TypeError(f"Query_llm must return str, got {type(raw_output)}")

    return validate_significance_with_one_repair(
        entities=entities,
        raw_output=raw_output,
        model_name=model_name,
        token_path=token_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )