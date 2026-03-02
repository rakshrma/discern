#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# -------------------------
# Strict schema
# -------------------------
ALLOWED_KEYS = ("history", "technique", "comparison", "findings", "impression", "others")


class ReportSections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: StrictStr = Field(default="")
    technique: StrictStr = Field(default="")
    comparison: StrictStr = Field(default="")
    findings: StrictStr = Field(default="")
    impression: StrictStr = Field(default="")
    others: StrictStr = Field(default="")


# -------------------------
# Prompts (YAML)
# -------------------------
def load_prompts_from_yaml(path: str) -> Dict[str, str]:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    required = ("system_prompt", "user_template", "repair_instructions")
    if not isinstance(data, dict) or any(k not in data for k in required):
        missing = [k for k in required if not isinstance(data, dict) or k not in data]
        raise ValueError(f"Bad YAML {path}. Missing: {missing}")
    for k in required:
        if not isinstance(data[k], str):
            raise ValueError(f"YAML key '{k}' must be a string.")
    return {k: data[k] for k in required}


def build_messages(system_prompt: str, user_template: str, report_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(report_text=report_text.strip())},
    ]


def build_repair_messages(system_prompt: str, repair_instructions: str, report_text: str, bad_output: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{repair_instructions}\n\n"
                f"REPORT:\n{report_text.strip()}\n\n"
                f"BAD_OUTPUT:\n{bad_output.strip()}\n\n"
                "Return ONLY valid JSON:"
            ),
        },
    ]


# -------------------------
# JSON extraction & cleanup
# -------------------------
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_first_json_object(text: str) -> Optional[str]:
    s = text.find("{")
    if s == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(s, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[s : i + 1]
    return None


def sanitize_json_text(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return _TRAILING_COMMA_RE.sub(r"\1", s).strip()


def coerce_to_schema(obj: Dict[str, Any]) -> Dict[str, str]:
    return {k: ("" if obj.get(k) is None else str(obj.get(k))) for k in ALLOWED_KEYS}


# -------------------------
# HF chat model runner
# -------------------------
@dataclass
class LLMConfig:
    model_name_or_path: str
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class HFChatLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        ).eval()

    @torch.inference_mode()
    def generate(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = {k: v.to(self.model.device) for k, v in self.tokenizer(prompt, return_tensors="pt").items()}

        do_sample = self.cfg.temperature > 0.0
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=self.cfg.temperature if do_sample else None,
            top_p=self.cfg.top_p if do_sample else None,
            repetition_penalty=self.cfg.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# -------------------------
# Parse with repair
# -------------------------
def _parse_or_raise(raw: str) -> ReportSections:
    cand = sanitize_json_text(extract_first_json_object(raw) or raw)
    obj = json.loads(cand)
    if not isinstance(obj, dict):
        raise ValueError("json_not_object")
    return ReportSections.model_validate(coerce_to_schema(obj))


def parse_single_report(
    llm: HFChatLLM,
    system_prompt: str,
    user_template: str,
    repair_instructions: str,
    report_text: str,
) -> Tuple[str, str, ReportSections]:
    raw1 = llm.generate(build_messages(system_prompt, user_template, report_text))
    try:
        return "ok", raw1, _parse_or_raise(raw1)
    except Exception as e:
        logger.debug("Pass1 failed: %r", e)

    raw2 = llm.generate(build_repair_messages(system_prompt, repair_instructions, report_text, raw1))
    return "repaired", raw2, _parse_or_raise(raw2)


# -------------------------
# Public API
# -------------------------
def run_single_report(
    *,
    model: str,
    prompt_yaml: str,
    report_text: str,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> Tuple[str, str, ReportSections]:
    if not isinstance(report_text, str) or not report_text.strip():
        raise ValueError("Missing report text.")

    p = load_prompts_from_yaml(prompt_yaml)
    llm = HFChatLLM(
        LLMConfig(
            model_name_or_path=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    )

    try:
        return parse_single_report(
            llm,
            system_prompt=p["system_prompt"],
            user_template=p["user_template"],
            repair_instructions=p["repair_instructions"],
            report_text=report_text,
        )
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        raise RuntimeError(f"Parsing failed: {e!r}") from e
