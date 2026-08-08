"""Subprocess entry point for CRIMSON scoring (run inside the crimson conda env).

Uses the official CRIMSON package API:

    from CRIMSON import CRIMSONScore
    scorer = CRIMSONScore()
    # batched (HF backend):
    results = scorer.evaluate_batch(refs, preds, batch_size=N)
    # per-pair (any backend):
    r = scorer.evaluate(reference_findings=..., predicted_findings=...)

The HF backend supports true batched generation via `evaluate_batch`. We chunk
the inputs, retry each chunk up to 3× with exponential backoff, and fall back
to per-pair `evaluate` for any chunk that still fails — same pattern as
CRIMSON/evaluate_reports.py in the upstream repo.

Note: the package import name is `CRIMSON` (uppercase) to avoid colliding with
this repo's wrapper module at src/metrics/crimson.py (lowercase) when both
are visible on sys.path.
"""
import json
import sys
import time


def main():
    _, input_path, output_path = sys.argv
    with open(input_path) as f:
        payload = json.load(f)

    refs       = payload["references"]
    cands      = payload["candidates"]
    batch_size = max(1, int(payload.get("batch_size", 8)))

    from CRIMSON import CRIMSONScore
    scorer = CRIMSONScore()

    n = len(refs)
    scores = [None] * n
    use_batch = getattr(scorer, "api", None) in ("huggingface", "hf")

    def _per_pair(start, r_chunk, c_chunk):
        for j, (ref, cand) in enumerate(zip(r_chunk, c_chunk)):
            try:
                r = scorer.evaluate(reference_findings=ref, predicted_findings=cand)
                scores[start + j] = float(r["crimson_score"])
            except Exception as exc:
                print(f"[CRIMSON] pair {start + j} failed: {exc}", file=sys.stderr)

    if use_batch:
        print(f"[CRIMSON] HF backend: batched inference (batch_size={batch_size})",
              file=sys.stderr)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            r_chunk, c_chunk = refs[start:end], cands[start:end]

            chunk_result = None
            for attempt in range(3):
                try:
                    chunk_result = scorer.evaluate_batch(
                        r_chunk, c_chunk, batch_size=batch_size,
                    )
                    break
                except Exception as exc:
                    print(f"[CRIMSON] batch {start} attempt {attempt} failed: {exc}",
                          file=sys.stderr)
                    if attempt < 2:
                        time.sleep(2 ** attempt)

            if chunk_result is None:
                _per_pair(start, r_chunk, c_chunk)
                continue
            for j, r in enumerate(chunk_result):
                if r is not None:
                    try:
                        scores[start + j] = float(r["crimson_score"])
                    except Exception as exc:
                        print(f"[CRIMSON] pair {start + j} parse failed: {exc}",
                              file=sys.stderr)

        # Per-pair retry for any pair that came back None (typically a JSON
        # parse failure inside evaluate_batch). Single-pair calls give the
        # model the full token budget and avoid batched-decoding artifacts.
        missing = [i for i, s in enumerate(scores) if s is None]
        if missing:
            print(f"[CRIMSON] retrying {len(missing)} unscored pair(s) individually",
                  file=sys.stderr)
            for i in missing:
                _per_pair(i, [refs[i]], [cands[i]])
    else:
        print(f"[CRIMSON] non-HF backend: per-pair scoring", file=sys.stderr)
        _per_pair(0, refs, cands)

    with open(output_path, "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()
