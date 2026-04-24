#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, RootModel, StrictStr

from call_llm import query_llm, get_token_usage, reset_token_usage, LLMConfig
from utils import sanitize_json_text, extract_first_json_value


# -------------------------
# Input contract
# -------------------------
class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_text: StrictStr
    prompt_yaml_path: StrictStr
    entities_yaml_path: StrictStr


# -------------------------
# YAML schemas
# -------------------------
class PromptYAML(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: StrictStr


class EntitiesYAML(RootModel[Dict[StrictStr, List[StrictStr]]]):
    @property
    def entity_pairs(self) -> List[Tuple[str, str]]:
        return [(str(cat), str(ent)) for cat, ents in self.root.items() for ent in ents]


def _load_yaml(path: str) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_prompt_yaml(path: str) -> PromptYAML:
    return PromptYAML.model_validate(_load_yaml(path))


def load_entities_yaml(path: str) -> EntitiesYAML:
    return EntitiesYAML.model_validate(_load_yaml(path))


# -------------------------
# Output schema (STRICT)
# -------------------------
Presence = Literal["POSITIVE", "NEGATIVE", "UNCERTAIN"]


class EntityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity: StrictStr
    presence: Presence
    sentence: StrictStr


class ExtractionOutput(RootModel[List[EntityDecision]]):
    """Output MUST be a JSON array of EntityDecision."""


# -------------------------
# JSON extraction helpers
# -------------------------

def extract_first_json_array(text: str) -> Optional[str]:
    return extract_first_json_value(text)


# -------------------------
# Prompt building
# -------------------------
def build_messages(prompt_template: str, entity_pairs: List[Tuple[str, str]], report_text: str) -> List[Dict[str, str]]:
    entity_list = "\n".join(f"- {cat} :: {ent}" for cat, ent in entity_pairs)
    system_prompt = prompt_template.format(entity_list=entity_list)
    user_prompt = f"Extract entities for the following report:\n\n{report_text}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_repair_messages(
    *,
    entity_pairs: List[Tuple[str, str]],
    report_text: str,
    bad_output: str,
    error_msg: str,
) -> List[Dict[str, str]]:
    allowed = "\n".join(f"- {c} :: {e}" for c, e in entity_pairs)
    user = (
        "Your previous output was invalid.\n"
        "Return ONLY a JSON array (no prose, no markdown).\n\n"
        "Each element must be exactly:\n"
        '{"entity":"<EXACT Category :: Entity>","presence":"POSITIVE|NEGATIVE|UNCERTAIN","sentence":"..."}\n\n'
        "Rules:\n"
        "- Do NOT output NOT MENTIONED.\n"
        "- If not mentioned, OMIT it.\n"
        "- Use ONLY the allowed entities below (exact match).\n\n"
        f"Allowed entities:\n{allowed}\n\n"
        f"Report:\n{report_text.strip()}\n\n"
        f"Validation error:\n{error_msg}\n\n"
        f"Bad output:\n{bad_output.strip()}\n\n"
        "Return ONLY JSON."
    )
    return [
        {"role": "system", "content": "You output only strict JSON. No explanations."},
        {"role": "user", "content": user},
    ]


# -------------------------
# Post-parse validation (keep-first + filter invalid)
# -------------------------
def _build_entity_part_index(allowed_entities: Set[str]) -> Dict[str, str]:
    """Map the entity-name part (after ' :: ') to the canonical 'Cat :: Ent' string.
    Used to recover when the model omits the category prefix or uses the wrong one."""
    index: Dict[str, str] = {}
    for ae in allowed_entities:
        # entity part is everything after the last ' :: '
        entity_part = ae.split(" :: ", 1)[-1].strip()
        if entity_part not in index:
            index[entity_part] = ae
    return index


def validate_entities(out_list: List[Dict[str, Any]], allowed_entities: Set[str]) -> List[Dict[str, Any]]:
    entity_part_index = _build_entity_part_index(allowed_entities)
    seen: Set[str] = set()
    cleaned: List[Dict[str, Any]] = []

    for i, row in enumerate(out_list):
        ent = row.get("entity")
        if ent is None:
            print(f"[ERROR] Missing 'entity' key at index {i}. Skipping.")
            continue
        if ent not in allowed_entities:
            # Fallback: match by entity-name part only (strips any category prefix).
            # This recovers from the model returning "Lung Volume" instead of
            # "Anatomical :: Lung Volume", or using a wrong category prefix.
            ent_part = ent.split(" :: ", 1)[-1].strip()
            canonical = entity_part_index.get(ent_part)
            if canonical:
                print(f"[WARNING] Normalizing entity '{ent}' → '{canonical}' at index {i}.")
                row = {**row, "entity": canonical}
                ent = canonical
            else:
                print(f"[ERROR] Invalid entity at index {i}: {ent}. Skipping.")
                continue
        if ent in seen:
            print(f"[WARNING] Duplicate entity at index {i}: {ent}. Keeping first occurrence.")
            continue
        seen.add(ent)
        cleaned.append(row)

    return cleaned


# -------------------------
# LLM call + parse + validate
# -------------------------
def _call_parse_validate(
    msgs: List[Dict[str, str]],
    *,
    cfg: LLMConfig,
    allowed_entities: Set[str],
) -> Tuple[List[Dict[str, Any]], str]:
    raw = query_llm(
        messages=msgs,
        model=cfg.model,
        token_path=cfg.token_path,
        hf_token_path=cfg.hf_token_path,
        max_tokens=cfg.max_tokens,
    )
    print(get_token_usage())
    reset_token_usage()

    if not isinstance(raw, str):
        raise TypeError(f"query_llm must return str, got {type(raw)}")

    arr_text = sanitize_json_text(extract_first_json_array(raw) or raw)
    parsed = ExtractionOutput.model_validate(json.loads(arr_text))
    out_list = validate_entities([item.model_dump() for item in parsed.root], allowed_entities)
    return out_list, raw


# -------------------------
# Core workflow
# -------------------------
def run_entity_extraction(
    *,
    cfg: LLMConfig,
    prompt_yaml_path: str,
    entities_yaml_path: str,
    report_text: str,
    require_all_entities: bool = False,
    enable_repair: bool = True,
) -> List[Dict[str, Any]]:
    if not isinstance(report_text, str) or not report_text.strip():
        raise ValueError("report_text must be a non-empty string.")

    prompt_yaml = load_prompt_yaml(prompt_yaml_path)
    entities_yaml = load_entities_yaml(entities_yaml_path)

    entity_pairs = entities_yaml.entity_pairs
    allowed_entities = {f"{c} :: {e}" for c, e in entity_pairs}
    messages = build_messages(prompt_yaml.prompt, entity_pairs, report_text)

    raw = ""
    try:
        out, raw = _call_parse_validate(messages, cfg=cfg, allowed_entities=allowed_entities)
        return out
    except Exception as e:
        if not enable_repair:
            raise RuntimeError(f"Validation failed: {e!r}") from e

        repair_messages = build_repair_messages(
            entity_pairs=entity_pairs,
            report_text=report_text,
            bad_output=raw,
            error_msg=str(e),
        )
        try:
            out2, _ = _call_parse_validate(repair_messages, cfg=cfg, allowed_entities=allowed_entities)
            return out2
        except Exception as repair_exc:
            raise RuntimeError(f"Repair attempt also failed: {repair_exc!r}")