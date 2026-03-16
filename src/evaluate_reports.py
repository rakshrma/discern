from __future__ import annotations

from typing import Any, Dict, List

from extract_entities import run_entity_extraction
from generate_attributes import run_compare_workflow
from evaluate_significance import run_clinical_significance_workflow
from utils import merge_common_with_missing_extra, merge_attributes_with_significance
from get_discern_score import compute_discern_score


def run_evaluation(
    report_text: str,
    candidate_text: str,
    model: str,
    token_path: str,
    prompt_yaml_path: str,
    entities_yaml_path: str,
    attribute_prompt_path: str,
    significance_yaml_path: str,
) -> List[Dict[str, Any]]:
    """
    End-to-end evaluation:
      1) Extract entities from reference + candidate
      2) Compare attributes on intersection
      3) Merge intersection + missing extras (presence optional)
      4) Score clinical significance
      5) Merge attributes + significance into final per-entity output
    """

    def _extract_entities(text: str) -> List[Dict[str, Any]]:
        return run_entity_extraction(
            model=model,
            token_path=token_path,
            prompt_yaml_path=prompt_yaml_path,
            entities_yaml_path=entities_yaml_path,
            report_text=text,
            enable_repair=True,
        )

    # 1) Entity extraction (ref + cand)
    entity_ref = _extract_entities(report_text)
    entity_cand = _extract_entities(candidate_text)
    print("reference entities: ", entity_ref)
    print("candidate_entities: ", entity_cand)

    # 2) Attribute comparison on overlap
    attribute_comparison = run_compare_workflow(
        prompt_path=attribute_prompt_path,
        candidate_report=candidate_text,
        ground_truth_report=report_text,
        candidate_entities=entity_cand,
        ground_truth_entities=entity_ref,
        model_name=model,
        db_token=token_path,
    )

    print("discordance assignment: ",attribute_comparison)

    # 3) Merge overlap + missing/extras
    merged_entities = merge_common_with_missing_extra(
        candidate_entities=entity_cand,
        reference_entities=entity_ref,
        common_attributions=attribute_comparison,
        include_presence=False,
    )

    print("complete discordance attribution: ",attribute_comparison)

    # 4) Clinical significance
    significance = run_clinical_significance_workflow(
        ground_truth_report=report_text,
        entities=merged_entities,
        prompt_yaml_path=significance_yaml_path,
        model_name=model,
        max_tokens=5000,
        token_path=token_path,
    )

    print("significance attribution: ",significance)

    discern_evaluation = merge_attributes_with_significance(
        merged_attributes=merged_entities,
        significance_output=significance,
    )

    print("Final discern evaluation: ",discern_evaluation)

    discern_score = compute_discern_score(discern_evaluation)

    # print("Final Discern evaluation: ",discern_evaluation)

    # 5) Final merge
    return discern_evaluation, discern_score