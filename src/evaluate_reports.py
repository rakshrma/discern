from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import time
from extract_entities import (
    run_entity_extraction,
    load_prompt_yaml as _load_entity_prompt_yaml,
    load_entities_yaml as _load_entities_yaml,
    build_messages as _build_entity_messages,
    build_repair_messages as _build_entity_repair_messages,
    sanitize_json_text,
    extract_first_json_array,
    ExtractionOutput,
    validate_entities,
)
from generate_attributes import (
    run_compare_workflow,
    load_prompt as _load_attr_prompt,
    entities_from_list_of_dicts,
    intersect_preserve_order,
    render_prompt as _render_attr_prompt,
    validate_with_one_repair as _validate_attr,
    ALL_PRESENCE,
)
from evaluate_significance import (
    run_clinical_significance_workflow,
    _load_prompt_yaml as _load_sig_prompt_yaml,
    build_messages as _build_sig_messages,
    validate_significance_with_one_repair as _validate_sig,
)
from call_llm import query_llm_batch, LLMConfig
from utils import merge_common_with_missing_extra, merge_attributes_with_significance
from get_discern_score import compute_reads_score


def run_evaluation(
    report_text: str,
    candidate_text: str,
    cfg: LLMConfig,
    prompt_yaml_path: str,
    entities_yaml_path: str,
    attribute_prompt_path: str,
    significance_yaml_path: str,
) -> Tuple[List[Dict[str, Any]], float]:
    def _extract_entities(text: str) -> List[Dict[str, Any]]:
        return run_entity_extraction(
            cfg=cfg,
            prompt_yaml_path=prompt_yaml_path,
            entities_yaml_path=entities_yaml_path,
            report_text=text,
            enable_repair=True,
        )

    start = time.time()
    entity_ref = _extract_entities(report_text)
    entity_cand = _extract_entities(candidate_text)
    end1 = time.time()
    print(f"Entity extraction time taken: {end1 - start:.2f}s")

    attribute_comparison = run_compare_workflow(
        prompt_path=attribute_prompt_path,
        candidate_report=candidate_text,
        ground_truth_report=report_text,
        candidate_entities=entity_cand,
        ground_truth_entities=entity_ref,
        cfg=cfg,
    )
    end2 = time.time()
    print(f"Attribute comparison time taken: {end2 - end1:.2f}s")

    merged_entities = merge_common_with_missing_extra(
        candidate_entities=entity_cand,
        reference_entities=entity_ref,
        common_attributions=attribute_comparison,
        include_presence=False,
    )
    end3 = time.time()
    print(f"Entity merge time taken: {end3 - end2:.2f}s")

    significance = run_clinical_significance_workflow(
        ground_truth_report=report_text,
        entities=merged_entities,
        prompt_yaml_path=significance_yaml_path,
        cfg=cfg,
    )
    end4 = time.time()
    print(f"Significance evaluation time taken: {end4 - end3:.2f}s")

    reads_evaluation = merge_attributes_with_significance(
        merged_attributes=merged_entities,
        significance_output=significance,
    )
    end5 = time.time()
    print(f"Final merge time taken: {end5 - end4:.2f}s")

    discern_score = compute_reads_score(reads_evaluation)
    end6 = time.time()
    print(f"Score calculation time taken: {end6 - end5:.2f}s")
    print(f"Total pipeline time: {end6 - start:.2f}s")

    return reads_evaluation, discern_score


def run_evaluation_batch(
    pairs: List[Tuple[str, str]],
    cfg: LLMConfig,
    prompt_yaml_path: str,
    entities_yaml_path: str,
    attribute_prompt_path: str,
    significance_yaml_path: str,
    batch_size: int = 100,
) -> List[Optional[Tuple[List[Dict[str, Any]], float]]]:
    """
    Batch version of run_evaluation.  Processes pairs stage-by-stage in chunks
    of `batch_size`, submitting each stage as one query_llm_batch call for
    GPU-efficient vLLM continuous batching.

    Returns a list of length N: each element is either
      (reads_evaluation, discern_score)  on success, or
      None                               if that pair failed at any stage.
    """
    N = len(pairs)
    if N == 0:
        return []

    if N > batch_size:
        results: List[Optional[Tuple[List[Dict[str, Any]], float]]] = []
        n_chunks = (N + batch_size - 1) // batch_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * batch_size
            chunk = pairs[start:start + batch_size]
            print(f"\n[DISCERN batch] Chunk {chunk_idx + 1}/{n_chunks} "
                  f"(samples {start}-{start + len(chunk) - 1})")
            results.extend(run_evaluation_batch(
                pairs=chunk,
                cfg=cfg,
                prompt_yaml_path=prompt_yaml_path,
                entities_yaml_path=entities_yaml_path,
                attribute_prompt_path=attribute_prompt_path,
                significance_yaml_path=significance_yaml_path,
                batch_size=batch_size,
            ))
        return results

    entity_prompt_yaml = _load_entity_prompt_yaml(prompt_yaml_path)
    entities_yaml      = _load_entities_yaml(entities_yaml_path)
    entity_pairs       = entities_yaml.entity_pairs
    allowed_entities   = {f"{c} :: {e}" for c, e in entity_pairs}

    attr_template    = _load_attr_prompt(attribute_prompt_path)
    sig_system_prompt = _load_sig_prompt_yaml(significance_yaml_path).prompt

    alive  = [True] * N
    errors: List[Optional[str]] = [None] * N

    def _batch_call(messages_list: List[List[Dict[str, str]]]) -> List[str]:
        return query_llm_batch(
            messages_batch=messages_list,
            model=cfg.model,
            token_path=cfg.token_path,
            hf_token_path=cfg.hf_token_path,
            max_tokens=cfg.max_tokens,
            temperature=0.0,
        )

    def _parse_entity_raw(raw: str, allowed: set) -> List[Dict]:
        arr = sanitize_json_text(extract_first_json_array(raw) or raw)
        parsed = ExtractionOutput.model_validate(json.loads(arr))
        return validate_entities([x.model_dump() for x in parsed.root], allowed)

    # Stages 1+2: entity extraction for ref and cand in one batch
    print(f"[DISCERN batch] Stages 1+2/4: entity extraction (ref + cand) — {2*N} prompts")
    t0 = time.time()
    all_entity_msgs = (
        [_build_entity_messages(entity_prompt_yaml.prompt, entity_pairs, ref) for ref, _ in pairs]
        + [_build_entity_messages(entity_prompt_yaml.prompt, entity_pairs, cand) for _, cand in pairs]
    )
    all_entity_raws = _batch_call(all_entity_msgs)
    ref_raws  = all_entity_raws[:N]
    cand_raws = all_entity_raws[N:]

    entity_ref:  List[Optional[List[Dict]]] = [None] * N
    entity_cand: List[Optional[List[Dict]]] = [None] * N

    ref_repair_needed:  List[Tuple[int, str, Exception]] = []
    cand_repair_needed: List[Tuple[int, str, Exception]] = []

    for i, raw in enumerate(ref_raws):
        try:
            entity_ref[i] = _parse_entity_raw(raw, allowed_entities)
        except Exception as e:
            ref_repair_needed.append((i, raw, e))

    for i, raw in enumerate(cand_raws):
        try:
            entity_cand[i] = _parse_entity_raw(raw, allowed_entities)
        except Exception as e:
            cand_repair_needed.append((i, raw, e))

    repair_msgs_all = (
        [_build_entity_repair_messages(entity_pairs=entity_pairs, report_text=pairs[i][0],
                                       bad_output=raw, error_msg=str(e))
         for i, raw, e in ref_repair_needed]
        + [_build_entity_repair_messages(entity_pairs=entity_pairs, report_text=pairs[i][1],
                                         bad_output=raw, error_msg=str(e))
           for i, raw, e in cand_repair_needed]
    )
    if repair_msgs_all:
        repair_raws = _batch_call(repair_msgs_all)
        n_ref = len(ref_repair_needed)
        for k, (i, _, _) in enumerate(ref_repair_needed):
            try:
                entity_ref[i] = _parse_entity_raw(repair_raws[k], allowed_entities)
            except Exception as re2:
                alive[i] = False
                errors[i] = f"entity_ref repair failed: {re2}"
        for k, (i, _, _) in enumerate(cand_repair_needed):
            try:
                entity_cand[i] = _parse_entity_raw(repair_raws[n_ref + k], allowed_entities)
            except Exception as re2:
                alive[i] = False
                errors[i] = f"entity_cand repair failed: {re2}"

    print(f"  done in {time.time()-t0:.1f}s  ({sum(alive)}/{N} alive)")

    # Stage 3: attribute comparison
    print(f"[DISCERN batch] Stage 3/4: attribute comparison — {sum(alive)} pairs")
    t0 = time.time()

    intersections: List[Optional[List[str]]] = [None] * N
    attr_msgs_idx: List[int] = []
    attr_msgs: List[List[Dict]] = []

    for i in range(N):
        if not alive[i]:
            continue
        cand_ents = entities_from_list_of_dicts(entity_cand[i], keep_presence=ALL_PRESENCE)
        gt_ents   = entities_from_list_of_dicts(entity_ref[i],  keep_presence=ALL_PRESENCE)
        isect = intersect_preserve_order(gt_ents, cand_ents)
        intersections[i] = isect
        if isect:
            ref_text, cand_text = pairs[i]
            attr_msgs_idx.append(i)
            attr_msgs.append(
                _render_attr_prompt(attr_template, ref_text.strip(), cand_text.strip(), isect)
            )

    attr_raws: List[Optional[str]] = [None] * N
    if attr_msgs:
        batch_raws = _batch_call(attr_msgs)
        for j, i in enumerate(attr_msgs_idx):
            attr_raws[i] = batch_raws[j]

    attr_results: List[Optional[List[Dict]]] = [None] * N
    for i in range(N):
        if not alive[i]:
            continue
        isect = intersections[i]
        if not isect:
            attr_results[i] = []
            continue
        try:
            attr_results[i] = _validate_attr(
                entities_intersection=isect,
                raw_response=attr_raws[i],
                cfg=cfg,
            )
        except Exception as e:
            alive[i] = False
            errors[i] = f"attribute comparison failed: {e}"
    print(f"  done in {time.time()-t0:.1f}s  ({sum(alive)}/{N} alive)")

    # Stage 4: clinical significance
    print(f"[DISCERN batch] Stage 4/4: clinical significance — {sum(alive)} pairs")
    t0 = time.time()

    merged_entities_list: List[Optional[List[Dict]]] = [None] * N
    sig_msgs_idx: List[int] = []
    sig_msgs: List[List[Dict]] = []

    for i in range(N):
        if not alive[i]:
            continue
        merged = merge_common_with_missing_extra(
            candidate_entities=entity_cand[i],
            reference_entities=entity_ref[i],
            common_attributions=attr_results[i],
            include_presence=False,
        )
        merged_entities_list[i] = merged
        if merged:
            ref_text, _ = pairs[i]
            sig_msgs_idx.append(i)
            sig_msgs.append(_build_sig_messages(sig_system_prompt, ref_text, merged))

    sig_raws: List[Optional[str]] = [None] * N
    if sig_msgs:
        batch_raws = _batch_call(sig_msgs)
        for j, i in enumerate(sig_msgs_idx):
            sig_raws[i] = batch_raws[j]

    sig_results: List[Optional[Dict]] = [None] * N
    for i in range(N):
        if not alive[i]:
            continue
        merged = merged_entities_list[i]
        if not merged:
            sig_results[i] = {}
            continue
        try:
            sig_results[i] = _validate_sig(
                entities=merged,
                raw_output=sig_raws[i],
                cfg=cfg,
            )
        except Exception as e:
            alive[i] = False
            errors[i] = f"significance scoring failed: {e}"
    print(f"  done in {time.time()-t0:.1f}s  ({sum(alive)}/{N} alive)")

    results: List[Optional[Tuple[List[Dict[str, Any]], float]]] = [None] * N
    for i in range(N):
        if not alive[i]:
            print(f"  [DISCERN batch] sample {i} FAILED: {errors[i]}")
            continue
        try:
            merged = merged_entities_list[i]
            if not merged:
                results[i] = ([], float(compute_reads_score([])))
                continue
            reads_eval = merge_attributes_with_significance(
                merged_attributes=merged,
                significance_output=sig_results[i],
            )
            score = compute_reads_score(reads_eval)
            results[i] = (reads_eval, score)
        except Exception as e:
            errors[i] = f"final merge/score failed: {e}"
            print(f"  [DISCERN batch] sample {i} FAILED: {errors[i]}")

    n_ok = sum(r is not None for r in results)
    print(f"[DISCERN batch] Complete — {n_ok}/{N} succeeded")
    return results
