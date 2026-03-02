from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

Presence = Literal["POSITIVE", "NEGATIVE"]


def _norm_entity(s: str) -> str:
    return " ".join(s.split())


def _make_row(
    entity_str: str,
    discrepancy_type: Literal["missing_in_candidate", "extra_in_candidate"],
    ref_presence: Optional[str],
    cand_presence: Optional[str],
    ref_finding: Optional[str],
    cand_finding: Optional[str],
    *,
    include_presence: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "entity": entity_str,
        "reference_report_finding": ref_finding,
        "candidate_report_finding": cand_finding,
        "diagnosis_concordance": "not mentioned",
        "location_concordance": "not mentioned",
        "severity_concordance": "not mentioned",
        "temporal_comparison": "not mentioned",
        "discrepancy_type": discrepancy_type,
    }
    if include_presence:
        row["reference_presence"] = ref_presence
        row["candidate_presence"] = cand_presence
    return row


def merge_common_with_missing_extra(
    candidate_entities: List[Dict[str, Any]],
    reference_entities: List[Dict[str, Any]],
    common_attributions: List[Dict[str, Any]],
    *,
    include_presence: bool = True,
) -> List[Dict[str, Any]]:
    """
    Inputs:
      candidate_entities: [{'entity': str, 'presence': 'POSITIVE'|'NEGATIVE'|...}, ...]
      reference_entities: same schema
      common_attributions: list of attribution dicts for entities that are common to both

    Output:
      common_attributions + rows for:
        - missing_in_candidate: present in reference, absent in candidate
        - extra_in_candidate: present in candidate, absent in reference

    Notes:
      - Entity strings in output are EXACT (taken from the originating list).
      - Appended rows contain a 'discrepancy_type' field.
      - Concordance fields default to 'not mentioned' for synthetic rows.
    """
    cand_by_norm: Dict[str, Dict[str, Any]] = {_norm_entity(d["entity"]): d for d in candidate_entities}
    ref_by_norm: Dict[str, Dict[str, Any]] = {_norm_entity(d["entity"]): d for d in reference_entities}

    common_norm = {_norm_entity(d["entity"]) for d in common_attributions}
    missing_in_candidate = sorted(ref_by_norm.keys() - cand_by_norm.keys())
    extra_in_candidate = sorted(cand_by_norm.keys() - ref_by_norm.keys())

    out: List[Dict[str, Any]] = list(common_attributions)

    for ne in missing_in_candidate:
        if ne in common_norm:
            continue
        ref_item = ref_by_norm[ne]
        out.append(_make_row(
            entity_str=ref_item["entity"],
            discrepancy_type="missing_in_candidate",
            ref_presence=ref_item.get("presence"),
            cand_presence="NOT MENTIONED" if include_presence else None,
            ref_finding=ref_item.get("sentence"),
            cand_finding=None,
            include_presence=include_presence,
        ))

    for ne in extra_in_candidate:
        if ne in common_norm:
            continue
        cand_item = cand_by_norm[ne]
        out.append(_make_row(
            entity_str=cand_item["entity"],
            discrepancy_type="extra_in_candidate",
            ref_presence="NOT MENTIONED" if include_presence else None,
            cand_presence=cand_item.get("presence"),
            ref_finding=None,
            cand_finding=cand_item.get("sentence"),
            include_presence=include_presence,
        ))

    return out


def merge_attributes_with_significance(
    merged_attributes: List[Dict[str, Any]],
    significance_output: Dict[str, Any],
    *,
    strict: bool = True,
) -> List[Dict[str, Any]]:
    """
    Merge attribute comparison output with clinical significance scores.

    Args:
        merged_attributes:
            List of entity dictionaries with concordance attributes.
        significance_output:
            JSON dict containing:
                {
                  "scored_entities": [
                      {"entity": ..., "significance_score": ..., "rationale": ...}
                  ]
                }
        strict:
            If True, raises error on missing/extra entities.
            If False, merges only matches and ignores extras.

    Returns:
        List[Dict] with significance fields appended.
    """
    if "scored_entities" not in significance_output:
        raise ValueError("significance_output must contain key 'scored_entities'.")

    score_lookup: Dict[str, Dict[str, Any]] = {
        item["entity"]: {
            "significance_score": item.get("significance_score"),
            "rationale": item.get("rationale"),
        }
        for item in significance_output["scored_entities"]
    }

    if strict:
        attr_entities = {item["entity"] for item in merged_attributes}
        score_entities = set(score_lookup.keys())
        missing_scores = attr_entities - score_entities
        extra_scores = score_entities - attr_entities
        if missing_scores:
            raise ValueError(f"Missing significance scores for: {missing_scores}")
        if extra_scores:
            raise ValueError(f"Extra significance scores for unknown entities: {extra_scores}")

    return [
        {**item, **score_lookup[item["entity"]]}
        if item["entity"] in score_lookup
        else {**item, "significance_score": None, "rationale": None}
        for item in merged_attributes
    ]