#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from call_llm import query_llm

# -------------------------
# Schemas
# -------------------------
Diag = Literal["concordant", "partial", "discordant"]
Conc = Literal["concordant", "partial", "discordant", "not mentioned", "candidate-adds", "candidate-misses"]
Temp = Literal["concordant", "partial", "candidate-adds", "candidate-misses", "discordant", "not mentioned"]
PresenceLabel = Literal["POSITIVE", "NEGATIVE", "UNCERTAIN"]

ALL_PRESENCE: List[PresenceLabel] = ["POSITIVE", "NEGATIVE", "UNCERTAIN"]


class EntityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity: str
    presence: PresenceLabel
    sentence: str


EntityListAdapter = TypeAdapter(List[EntityItem])


class CompareRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity: str
    reference_report_finding: str
    candidate_report_finding: str
    diagnosis_concordance: Diag
    location_concordance: Conc
    severity_concordance: Conc
    temporal_comparison: Temp


_COMPARE_ROW_SCHEMA = CompareRow.model_json_schema()

# -------------------------
# Inputs
# -------------------------
def load_prompt(prompt_path: str) -> str:
    return Path(prompt_path).read_text(encoding="utf-8")


def _resolve_to_obj(obj: Any) -> Any:
    if not isinstance(obj, str):
        return obj
    s = obj.strip()
    p = Path(s)
    if p.is_file():
        s = p.read_text(encoding="utf-8").strip()
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError("String input must be a JSON list or a path to a JSON file.") from e


def entities_from_list_of_dicts(
    obj: Any,
    *,
    keep_presence: Union[PresenceLabel, List[PresenceLabel], None] = None,
) -> List[str]:
    items = EntityListAdapter.validate_python(_resolve_to_obj(obj))
    keep_set = None if keep_presence is None else (set(keep_presence) if isinstance(keep_presence, list) else {keep_presence})
    out, seen = [], set()
    for it in items:
        if keep_set is not None and it.presence not in keep_set:
            continue
        if it.entity not in seen:
            seen.add(it.entity)
            out.append(it.entity)
    return out


def intersect_preserve_order(a: Sequence[str], b: Sequence[str]) -> List[str]:
    bset = {x.strip() for x in b if x and x.strip()}
    out, seen = [], set()
    for x in a:
        x = (x or "").strip()
        if x and x in bset and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -------------------------
# Prompt rendering
# -------------------------
def render_prompt(system_template: str, ref_report: str, pred_report: str, entity_list: List[str]) -> List[Dict[str, str]]:
    system_prompt = system_template.replace("{ entity_list_json }", json.dumps(entity_list, indent=2))
    user_prompt = (
        "Evaluate the following reports.\n\n"
        "Ground Truth Report:\n"
        f"{ref_report}\n\n"
        "Candidate Report:\n"
        f"{pred_report}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


# -------------------------
# LLM output parsing/validation + one-shot repair
# -------------------------
def extract_first_json_array(text: str) -> str:
    m = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON array detected in model response.")
    return m.group(1).strip()


def _dedup_and_sort(rows: List[CompareRow], ordered: List[str]) -> List[CompareRow]:
    idx = {e: i for i, e in enumerate(ordered)}
    seen, out = set(), []
    for r in rows:
        if r.entity in idx and r.entity not in seen:
            seen.add(r.entity)
            out.append(r)
    out.sort(key=lambda r: idx[r.entity])
    return out


def validate_rows(expected_entities: List[str], json_array_str: str) -> List[dict]:
    data = json.loads(json_array_str)
    if not isinstance(data, list):
        raise ValueError("Output is not a JSON array.")

    rows = [CompareRow.model_validate(item) for item in data]
    exp_set = set(expected_entities)
    got_set = {r.entity for r in rows}

    extra = sorted(got_set - exp_set)
    if extra:
        raise ValueError(f"Model returned entities not in provided intersection list: {extra}")

    missing = [e for e in expected_entities if e not in got_set]
    if missing:
        raise ValueError(f"Model omitted entities that must be included: {missing}")

    return [r.model_dump() for r in _dedup_and_sort(rows, expected_entities)]


def build_repair_messages(
    *,
    entities_intersection: List[str],
    error_msg: str,
    raw_response: str,
) -> List[Dict[str, str]]:
    sys = (
        "You are a strict JSON repair assistant.\n"
        "Return ONLY a valid JSON array (no markdown, no commentary).\n"
        "Follow the provided JSON Schema for each item.\n"
        "Include EXACTLY the entities from the entity list (no extras, no missing), once each, in that order.\n"
        "If duplicates exist, keep only the first occurrence.\n"
    )
    usr = (
        "Fix the model output to satisfy all constraints.\n\n"
        "Entity list:\n"
        f"{json.dumps(entities_intersection, indent=2)}\n\n"
        "JSON Schema:\n"
        f"{json.dumps(_COMPARE_ROW_SCHEMA, indent=2)}\n\n"
        "Validation error:\n"
        f"{error_msg}\n\n"
        "Original model output:\n"
        f"{raw_response}\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def validate_with_one_repair(
    *,
    entities_intersection: List[str],
    raw_response: str,
    model_name: str,
    token_path: str,
    token_size: int,
) -> List[dict]:
    try:
        return validate_rows(entities_intersection, extract_first_json_array(raw_response))
    except (ValueError, ValidationError, JSONDecodeError) as e:
        repaired = query_llm(
            model=model_name,
            token_path=token_path,
            max_tokens=token_size,
            messages=build_repair_messages(
                entities_intersection=entities_intersection,
                error_msg=str(e),
                raw_response=raw_response,
            ),
        )
        return validate_rows(entities_intersection, extract_first_json_array(repaired))


# -------------------------
# Main workflow
# -------------------------
def run_compare_workflow(
    prompt_path: str,
    candidate_report: str,
    ground_truth_report: str,
    candidate_entities: Union[str, List[str]],
    ground_truth_entities: Union[str, List[str]],
    model_name: str,
    db_token: str,
    token_size: int = 5000,
    temperature: float = 0.0,
) -> List[dict]:
    template = load_prompt(prompt_path)

    cand = entities_from_list_of_dicts(candidate_entities, keep_presence=ALL_PRESENCE)
    gt = entities_from_list_of_dicts(ground_truth_entities, keep_presence=ALL_PRESENCE)

    entities_intersection = intersect_preserve_order(gt, cand)
    if not entities_intersection:
        return []

    messages = render_prompt(template, ground_truth_report.strip(), candidate_report.strip(), entities_intersection)

    raw = query_llm(model=model_name, token_path=db_token, max_tokens=token_size, messages=messages, temperature=temperature)

    return validate_with_one_repair(
        entities_intersection=entities_intersection,
        raw_response=raw,
        model_name=model_name,
        token_path=db_token,
        token_size=token_size,
    )