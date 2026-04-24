"""
Radiology report concordance evaluator.

Compares a candidate report against a reference (ground truth) report
at the entity level using an LLM, with Pydantic-validated output.
"""

from __future__ import annotations

import json
import re
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, ValidationError

from call_llm import (query_llm, query_llm_batch,
                      TokenLimitError, TOKEN_LIMIT_PREFIX)  # adjust import path as needed

MAX_TOKENS_CAP = 25000   # never escalate beyond model hard limit

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────

class PresenceStatus(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNCERTAIN = "UNCERTAIN"
    NOT_MENTIONED = "NOT MENTIONED"


DiagnosisConcordance = Literal[
    "concordant", "partial", "discordant", "candidate-adds", "candidate-misses"
]

DimensionConcordance = Literal[
    "concordant", "partial", "discordant", "not mentioned",
    "candidate-adds", "candidate-misses",
]


class EntityEvaluation(BaseModel):
    entity_name: str
    reference_presence: PresenceStatus
    candidate_presence: PresenceStatus
    reference_text: Optional[str] = None
    candidate_text: Optional[str] = None
    diagnosis_concordance: DiagnosisConcordance
    location_concordance: DimensionConcordance
    severity_concordance: DimensionConcordance
    temporal_concordance: DimensionConcordance
    clinical_significance_score: int = Field(ge=0, le=4)
    rationale: str

    @field_validator("clinical_significance_score")
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if not 0 <= v <= 4:
            raise ValueError(f"clinical_significance_score must be 0-4, got {v}")
        return v


class EvaluationResult(BaseModel):
    """Wrapper so we can validate the full LLM response as a list."""
    entities: List[EntityEvaluation]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return the parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_entity_list(raw: dict) -> List[str]:
    """
    Flatten a nested category → entity-list YAML into a flat list of
    entity names.

    Accepts formats like:
        Quality of Exams:
          - Suboptimal Penetration
          - Suboptimal Inspiration
        Tubes and Lines:
          - Endotracheal Tubes
          - Central Venous Catheters

    Returns: ["Suboptimal Penetration", "Suboptimal Inspiration",
              "Endotracheal Tubes", "Central Venous Catheters"]
    """
    entities: List[str] = []
    for category, items in raw.items():
        if isinstance(items, list):
            entities.extend(items)
        else:
            logger.warning(
                "Skipping category '%s': expected a list, got %s",
                category, type(items).__name__,
            )
    return entities


def _extract_json(text: str) -> str:
    """
    Extract a JSON array from LLM output that may contain markdown
    fences or surrounding prose.
    """
    # Try to find a JSON array between ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\[.*?])\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Try bare JSON array
    bare_match = re.search(r"(\[.*])", text, re.DOTALL)
    if bare_match:
        return bare_match.group(1)

    return text.strip()


def _build_messages(
    prompt_config: dict,
    reference_report: str,
    candidate_report: str,
    entity_list: List[str],
) -> List[Dict[str, str]]:
    """Construct the chat messages from the prompt template + inputs."""
    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_prompt_template"]

    entity_list_json = json.dumps(entity_list, indent=2)

    user_content = (
        user_template
        .replace("{reference_report}", reference_report)
        .replace("{candidate_report}", candidate_report)
        .replace("{entity_list_json}", entity_list_json)
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_retry_message(errors: str, raw_response: str) -> str:
    """Build a follow-up user message asking the LLM to fix validation errors."""
    return (
        "Your previous response failed schema validation with the following errors:\n\n"
        f"{errors}\n\n"
        "Please re-read the output format instructions and return the FULL corrected "
        "JSON array. Return ONLY the JSON — no explanations."
    )


def _validate_single_response(
    raw_response: str,
    idx: int = 0,
) -> Optional[List[EntityEvaluation]]:
    """Try to parse and validate a single LLM response. Returns None on failure."""
    try:
        json_str = _extract_json(raw_response)
        parsed = json.loads(json_str)
        if not isinstance(parsed, list):
            return None
        result = EvaluationResult(entities=parsed)
        return result.entities
    except (json.JSONDecodeError, ValueError, ValidationError) as e:
        logger.warning("Item %d failed validation: %s", idx, e)
        return None


# ──────────────────────────────────────────────
# Single evaluator
# ──────────────────────────────────────────────

def evaluate_reports(
    reference_report: str,
    candidate_report: str,
    entity_list_path: str | Path,
    prompt_path: str | Path,
    model: Optional[str] = None,
    token_path: Optional[str] = None,
    hf_token_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    max_retries: int = 2,
) -> List[EntityEvaluation]:
    """
    Evaluate entity-level concordance between a reference and candidate
    radiology report.

    Parameters
    ----------
    reference_report : str
        The ground truth radiology report text.
    candidate_report : str
        The candidate radiology report text to evaluate.
    entity_list_path : str or Path
        Path to a YAML file mapping categories to entity lists.
    prompt_path : str or Path
        Path to the prompt YAML file with ``system_prompt`` and
        ``user_prompt_template`` keys.
    model : str, optional
        Model name passed to ``query_llm``.
    token_path : str, optional
        Databricks token path passed to ``query_llm``.
    hf_token_path : str, optional
        HuggingFace token path passed to ``query_llm``.
    max_tokens : int
        Max tokens for LLM generation (default 4096).
    temperature : float
        Sampling temperature (default 0.1).
    max_retries : int
        Number of retry attempts on validation failure (default 2).

    Returns
    -------
    list[EntityEvaluation]
        Validated list of entity evaluations.

    Raises
    ------
    ValidationError
        If the LLM output cannot be parsed into valid schema after all retries.
    """
    # Load inputs
    prompt_config = _load_yaml(prompt_path)
    entity_config = _load_yaml(entity_list_path)
    entity_list: List[str] = _flatten_entity_list(entity_config)

    # Build initial messages
    messages = _build_messages(
        prompt_config, reference_report, candidate_report, entity_list
    )

    # Build kwargs for query_llm
    llm_kwargs: Dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        llm_kwargs["model"] = model
    if token_path:
        llm_kwargs["token_path"] = token_path
    if hf_token_path:
        llm_kwargs["hf_token_path"] = hf_token_path

    last_error: Optional[Exception] = None

    for attempt in range(1 + max_retries):
        # Call the LLM — may raise TokenLimitError
        try:
            raw_response = query_llm(**llm_kwargs)
        except TokenLimitError as e:
            last_error = e
            current_max = llm_kwargs["max_tokens"]
            new_max = min(int(current_max * 1.5), MAX_TOKENS_CAP)
            logger.warning(
                "Attempt %d/%d hit token limit (max_tokens=%d). "
                "Escalating to %d and retrying with fresh prompt.",
                attempt + 1, 1 + max_retries, current_max, new_max,
            )
            llm_kwargs["max_tokens"] = new_max
            # Reset to original messages (no conversation history)
            llm_kwargs["messages"] = _build_messages(
                prompt_config, reference_report, candidate_report, entity_list
            )
            continue

        # Extract and parse JSON
        try:
            json_str = _extract_json(raw_response)
            parsed = json.loads(json_str)

            if not isinstance(parsed, list):
                raise ValueError(
                    f"Expected a JSON array, got {type(parsed).__name__}"
                )

            # Validate each entity through Pydantic
            result = EvaluationResult(entities=parsed)

            logger.info(
                "Validation passed on attempt %d/%d — %d entities returned.",
                attempt + 1, 1 + max_retries, len(result.entities),
            )
            return result.entities

        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_error = e
            error_msg = str(e)
            logger.warning(
                "Attempt %d/%d failed validation: %s",
                attempt + 1, 1 + max_retries, error_msg,
            )

            if attempt < max_retries:
                # Append the failed response and a correction request
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": _build_retry_message(error_msg, raw_response),
                })
                # Update messages in kwargs for next call
                llm_kwargs["messages"] = messages

    # All retries exhausted
    raise last_error  # type: ignore[misc]


# ──────────────────────────────────────────────
# Batch evaluator
# ──────────────────────────────────────────────

def evaluate_reports_batch(
    reference_reports: List[str],
    candidate_reports: List[str],
    entity_list_path: str | Path,
    prompt_path: str | Path,
    model: Optional[str] = None,
    token_path: Optional[str] = None,
    hf_token_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    max_retries: int = 2,
    max_concurrent: int = 3,
) -> List[Optional[List[EntityEvaluation]]]:
    """
    Batch-evaluate entity-level concordance for multiple report pairs.

    Sends all pairs in a single batch call via query_llm_batch, then
    validates each response individually. Failed validations are retried
    sequentially (since retry needs conversational context).

    Parameters
    ----------
    reference_reports : list of str
        Ground truth report texts.
    candidate_reports : list of str
        Candidate report texts to evaluate.
    entity_list_path : str or Path
        Path to YAML entity list (nested category format).
    prompt_path : str or Path
        Path to the prompt YAML file.
    model : str, optional
        Model name passed to query_llm_batch.
    token_path : str, optional
        Databricks token path.
    hf_token_path : str, optional
        HuggingFace token path.
    max_tokens : int
        Max tokens per completion (default 4096).
    temperature : float
        Sampling temperature (default 0.1).
    max_retries : int
        Number of sequential retry attempts for failed validations (default 2).
    max_concurrent : int
        Max concurrent API requests for Databricks batch (default 5).

    Returns
    -------
    list of (list[EntityEvaluation] or None)
        One result per input pair. None if validation failed after all retries.
    """
    assert len(reference_reports) == len(candidate_reports), (
        f"Mismatched lengths: {len(reference_reports)} references vs "
        f"{len(candidate_reports)} candidates"
    )

    n = len(reference_reports)
    if n == 0:
        return []

    # Load configs once
    prompt_config = _load_yaml(prompt_path)
    entity_config = _load_yaml(entity_list_path)
    entity_list: List[str] = _flatten_entity_list(entity_config)

    # Build all message lists
    all_messages = [
        _build_messages(prompt_config, ref, cand, entity_list)
        for ref, cand in zip(reference_reports, candidate_reports)
    ]

    # Build batch kwargs
    batch_kwargs: Dict = {
        "messages_batch": all_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_concurrent": max_concurrent,
    }
    if model:
        batch_kwargs["model"] = model
    if token_path:
        batch_kwargs["token_path"] = token_path
    if hf_token_path:
        batch_kwargs["hf_token_path"] = hf_token_path

    # ── First pass: batch call ──
    print(f"[Batch] Sending {n} report pairs ...")
    raw_responses = query_llm_batch(**batch_kwargs)

    # ── Validate each response ──
    results: List[Optional[List[EntityEvaluation]]] = [None] * n
    failed_indices: List[int] = []

    for i, raw in enumerate(raw_responses):
        validated = _validate_single_response(raw, i)
        if validated is not None:
            results[i] = validated
        else:
            failed_indices.append(i)

    print(f"[Batch] First pass: {n - len(failed_indices)}/{n} validated, "
          f"{len(failed_indices)} need retry.")

    # ── Retry failures sequentially (need conversational context) ──
    if failed_indices and max_retries > 0:
        print(f"[Batch] Retrying {len(failed_indices)} failed items sequentially ...")

        # Build single-call kwargs template
        single_kwargs_base: Dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            single_kwargs_base["model"] = model
        if token_path:
            single_kwargs_base["token_path"] = token_path
        if hf_token_path:
            single_kwargs_base["hf_token_path"] = hf_token_path

        for i in failed_indices:
            messages = list(all_messages[i])  # copy original
            last_raw = raw_responses[i]
            current_max = max_tokens

            success = False
            for attempt in range(max_retries):
                # Token-limit hit in batch pass → escalate and retry with fresh prompt
                if last_raw.startswith(TOKEN_LIMIT_PREFIX):
                    current_max = min(int(current_max * 1.5), MAX_TOKENS_CAP)
                    print(f"  Item {i}: token limit hit, escalating to "
                          f"max_tokens={current_max}")
                    messages = list(all_messages[i])  # fresh prompt, no history
                else:
                    messages.append({"role": "assistant", "content": last_raw})
                    messages.append({
                        "role": "user",
                        "content": _build_retry_message(
                            f"Failed to parse valid JSON from response (attempt {attempt + 1})",
                            last_raw,
                        ),
                    })

                single_kwargs = {**single_kwargs_base,
                                 "messages": messages,
                                 "max_tokens": current_max}
                try:
                    last_raw = query_llm(**single_kwargs)
                except TokenLimitError:
                    current_max = min(int(current_max * 1.5), MAX_TOKENS_CAP)
                    last_raw = TOKEN_LIMIT_PREFIX  # will trigger escalation next loop
                    continue

                validated = _validate_single_response(last_raw, i)
                if validated is not None:
                    results[i] = validated
                    success = True
                    print(f"  Item {i}: validated on retry {attempt + 1} "
                          f"(max_tokens={current_max})")
                    break

            if not success:
                logger.error("Item %d: failed after %d retries.", i, max_retries)

    succeeded = sum(1 for r in results if r is not None)
    print(f"[Batch] Final: {succeeded}/{n} succeeded.")

    return results