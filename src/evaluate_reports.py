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
from call_llm import query_llm_batch
from utils import merge_common_with_missing_extra, merge_attributes_with_significance
from get_discern_score import compute_reads_score


def run_evaluation(
    report_text: str,
    candidate_text: str,
    model: str,
    token_path: str,
    prompt_yaml_path: str,
    entities_yaml_path: str,
    attribute_prompt_path: str,
    significance_yaml_path: str,
    max_tokens: int = 5000,
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
            max_tokens=max_tokens,
        )

    # 1) Entity extraction (ref + cand)
    start = time.time()
    entity_ref = _extract_entities(report_text)
    entity_cand = _extract_entities(candidate_text)
    end1 = time.time()
    print(f"Entity extraction time taken: {end1 - start:.2f}s")

    # 2) Attribute comparison on overlap
    attribute_comparison = run_compare_workflow(
        prompt_path=attribute_prompt_path,
        candidate_report=candidate_text,
        ground_truth_report=report_text,
        candidate_entities=entity_cand,
        ground_truth_entities=entity_ref,
        model_name=model,
        db_token=token_path,
        token_size=max_tokens,
    )
    end2 = time.time()
    print(f"Attribute comparison time taken: {end2 - end1:.2f}s")

    # 3) Merge overlap + missing/extras
    merged_entities = merge_common_with_missing_extra(
        candidate_entities=entity_cand,
        reference_entities=entity_ref,
        common_attributions=attribute_comparison,
        include_presence=False,
    )
    end3 = time.time()
    print(f"Entity merge time taken: {end3 - end2:.2f}s")

    # 4) Clinical significance
    significance = run_clinical_significance_workflow(
        ground_truth_report=report_text,
        entities=merged_entities,
        prompt_yaml_path=significance_yaml_path,
        model_name=model,
        max_tokens=max_tokens,
        token_path=token_path,
    )
    end4 = time.time()
    print(f"Significance evaluation time taken: {end4 - end3:.2f}s")

    # 5) Final merge
    reads_evaluation = merge_attributes_with_significance(
        merged_attributes=merged_entities,
        significance_output=significance,
    )
    end5 = time.time()
    print(f"Final merge time taken: {end5 - end4:.2f}s")

    # 6) Score calculation
    discern_score = compute_reads_score(reads_evaluation)
    end6 = time.time()
    print(f"Score calculation time taken: {end6 - end5:.2f}s")

    # Total
    print(f"Total pipeline time: {end6 - start:.2f}s")

    return reads_evaluation, discern_score


def run_evaluation_batch(
    pairs: List[Tuple[str, str]],
    model: str,
    token_path: str,
    prompt_yaml_path: str,
    entities_yaml_path: str,
    attribute_prompt_path: str,
    significance_yaml_path: str,
    hf_token_path: str = "",
    max_tokens: int = 5000,
    batch_size: int = 100,
) -> List[Optional[Tuple[List[Dict[str, Any]], float]]]:
    """
    Batch version of run_evaluation.  Processes pairs stage-by-stage in chunks
    of `batch_size`, submitting each stage as one query_llm_batch call for
    GPU-efficient vLLM continuous batching.

    Returns a list of length N: each element is either
      (reads_evaluation, discern_score)  — on success, or
      None                               — if that pair failed at any stage.
    """
    N = len(pairs)
    if N == 0:
        return []

    # Chunk into sub-batches and concatenate results
    if N > batch_size:
        results: List[Optional[Tuple[List[Dict[str, Any]], float]]] = []
        n_chunks = (N + batch_size - 1) // batch_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * batch_size
            chunk = pairs[start:start + batch_size]
            print(f"\n[DISCERN batch] Chunk {chunk_idx + 1}/{n_chunks} "
                  f"(samples {start}–{start + len(chunk) - 1})")
            results.extend(run_evaluation_batch(
                pairs=chunk,
                model=model,
                token_path=token_path,
                prompt_yaml_path=prompt_yaml_path,
                entities_yaml_path=entities_yaml_path,
                attribute_prompt_path=attribute_prompt_path,
                significance_yaml_path=significance_yaml_path,
                hf_token_path=hf_token_path,
                max_tokens=max_tokens,
                batch_size=batch_size,
            ))
        return results

    # ── Load config once ──────────────────────────────────────────────────
    entity_prompt_yaml  = _load_entity_prompt_yaml(prompt_yaml_path)
    entities_yaml       = _load_entities_yaml(entities_yaml_path)
    entity_pairs        = entities_yaml.entity_pairs
    allowed_entities    = {f"{c} :: {e}" for c, e in entity_pairs}

    attr_template       = _load_attr_prompt(attribute_prompt_path)
    sig_system_prompt   = _load_sig_prompt_yaml(significance_yaml_path).prompt

    hf_batch_kwargs: Dict[str, Any] = {"hf_token_path": hf_token_path} if hf_token_path else {}

    # alive[i] = True while sample i hasn't failed
    alive  = [True] * N
    errors: List[Optional[str]] = [None] * N

    def _batch_call(messages_list: List[List[Dict[str, str]]], max_tok: int) -> List[str]:
        return query_llm_batch(
            messages_batch=messages_list,
            model=model,
            token_path=token_path,
            max_tokens=max_tok,
            temperature=0.0,
            **hf_batch_kwargs,
        )

    # ── Stage 1: entity extraction — reference reports ────────────────────
    print(f"[DISCERN batch] Stage 1/4: entity extraction (reference) — {N} reports")
    t0 = time.time()
    ref_msgs = [
        _build_entity_messages(entity_prompt_yaml.prompt, entity_pairs, ref)
        for ref, _ in pairs
    ]
    ref_raws = _batch_call(ref_msgs, max_tokens)

    entity_ref: List[Optional[List[Dict]]] = [None] * N
    for i, raw in enumerate(ref_raws):
        try:
            arr = sanitize_json_text(extract_first_json_array(raw) or raw)
            parsed = ExtractionOutput.model_validate(json.loads(arr))
            entity_ref[i] = validate_entities(
                [x.model_dump() for x in parsed.root], allowed_entities
            )
        except Exception as e:
            # Repair: one additional individual call
            try:
                repair_msgs = _build_entity_repair_messages(
                    entity_pairs=entity_pairs,
                    report_text=pairs[i][0],
                    bad_output=raw,
                    error_msg=str(e),
                )
                repair_raw = query_llm_batch(
                    [repair_msgs], model=model, token_path=token_path,
                    max_tokens=max_tokens, temperature=0.0, **hf_batch_kwargs,
                )[0]
                arr2 = sanitize_json_text(extract_first_json_array(repair_raw) or repair_raw)
                parsed2 = ExtractionOutput.model_validate(json.loads(arr2))
                entity_ref[i] = validate_entities(
                    [x.model_dump() for x in parsed2.root], allowed_entities
                )
            except Exception as re2:
                alive[i] = False
                errors[i] = f"entity_ref repair failed: {re2}"
    print(f"  done in {time.time()-t0:.1f}s  ({sum(alive)}/{N} alive)")

    # ── Stage 2: entity extraction — candidate reports ────────────────────
    print(f"[DISCERN batch] Stage 2/4: entity extraction (candidate) — {N} reports")
    t0 = time.time()
    cand_msgs = [
        _build_entity_messages(entity_prompt_yaml.prompt, entity_pairs, cand)
        for _, cand in pairs
    ]
    cand_raws = _batch_call(cand_msgs, max_tokens)

    entity_cand: List[Optional[List[Dict]]] = [None] * N
    for i, raw in enumerate(cand_raws):
        if not alive[i]:
            continue
        try:
            arr = sanitize_json_text(extract_first_json_array(raw) or raw)
            parsed = ExtractionOutput.model_validate(json.loads(arr))
            entity_cand[i] = validate_entities(
                [x.model_dump() for x in parsed.root], allowed_entities
            )
        except Exception as e:
            try:
                repair_msgs = _build_entity_repair_messages(
                    entity_pairs=entity_pairs,
                    report_text=pairs[i][1],
                    bad_output=raw,
                    error_msg=str(e),
                )
                repair_raw = query_llm_batch(
                    [repair_msgs], model=model, token_path=token_path,
                    max_tokens=max_tokens, temperature=0.0, **hf_batch_kwargs,
                )[0]
                arr2 = sanitize_json_text(extract_first_json_array(repair_raw) or repair_raw)
                parsed2 = ExtractionOutput.model_validate(json.loads(arr2))
                entity_cand[i] = validate_entities(
                    [x.model_dump() for x in parsed2.root], allowed_entities
                )
            except Exception as re2:
                alive[i] = False
                errors[i] = f"entity_cand repair failed: {re2}"
    print(f"  done in {time.time()-t0:.1f}s  ({sum(alive)}/{N} alive)")

    # ── Stage 3: attribute comparison ────────────────────────────────────
    print(f"[DISCERN batch] Stage 3/4: attribute comparison — {sum(alive)} pairs")
    t0 = time.time()

    # Pre-compute intersections (no LLM needed)
    intersections: List[Optional[List[str]]] = [None] * N
    attr_msgs_idx: List[int] = []     # indices of samples that need attribute comparison
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
        batch_raws = _batch_call(attr_msgs, max_tokens)
        for j, i in enumerate(attr_msgs_idx):
            attr_raws[i] = batch_raws[j]

    attr_results: List[Optional[List[Dict]]] = [None] * N
    for i in range(N):
        if not alive[i]:
            continue
        isect = intersections[i]
        if not isect:
            attr_results[i] = []  # empty intersection → no attribute comparison
            continue
        raw = attr_raws[i]
        try:
            attr_results[i] = _validate_attr(
                entities_intersection=isect,
                raw_response=raw,
                model_name=model,
                token_path=token_path,
                token_size=max_tokens,
            )
        except Exception as e:
            alive[i] = False
            errors[i] = f"attribute comparison failed: {e}"
    print(f"  done in {time.time()-t0:.1f}s  ({sum(alive)}/{N} alive)")

    # ── Stage 4: clinical significance ────────────────────────────────────
    print(f"[DISCERN batch] Stage 4/4: clinical significance — {sum(alive)} pairs")
    t0 = time.time()

    # Local merge (no LLM needed)
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
        batch_raws = _batch_call(sig_msgs, max_tokens)
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
        raw = sig_raws[i]
        try:
            sig_results[i] = _validate_sig(
                entities=merged,
                raw_output=raw,
                model_name=model,
                token_path=token_path,
                max_tokens=max_tokens,
                temperature=0.1,
            )
        except Exception as e:
            alive[i] = False
            errors[i] = f"significance scoring failed: {e}"
    print(f"  done in {time.time()-t0:.1f}s  ({sum(alive)}/{N} alive)")

    # ── Final: local merge + score ────────────────────────────────────────
    results: List[Optional[Tuple[List[Dict[str, Any]], float]]] = [None] * N
    for i in range(N):
        if not alive[i]:
            print(f"  [DISCERN batch] sample {i} FAILED: {errors[i]}")
            continue
        try:
            merged = merged_entities_list[i]
            if not merged:
                # No entities after merge → score is 0, evaluation is empty
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