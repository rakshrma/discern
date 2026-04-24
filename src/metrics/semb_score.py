"""
SembScore — CheXbert embedding cosine similarity.
Matches the s_emb metric in CXR-Report-Metric / RadCliQ.

Requires the CheXbert model checkpoint. Download from:
https://stanfordmedicine.box.com/s/c3stck6w6dol0vbtr3m9pj63fas8b8or
Set checkpoints.chexbert_path in config.yaml.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _load_chexbert(chexbert_path: str):
    """Load CheXbert model from checkpoint directory."""
    from transformers import BertTokenizer
    import sys

    # CheXbert uses a custom model; we load via the CXR-Report-Metric approach
    # (cosine similarity of BERT [CLS] embeddings from CheXbert)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    ckpt = Path(chexbert_path)
    if ckpt.is_dir():
        ckpt_file = next(ckpt.glob("*.pth"), None) or next(ckpt.glob("*.bin"), None)
        if ckpt_file is None:
            raise FileNotFoundError(f"No .pth or .bin checkpoint found in {chexbert_path}")
        ckpt = ckpt_file

    from transformers import BertModel, BertConfig
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=14)
    model = BertModel(config)
    state = torch.load(str(ckpt), map_location="cpu")
    # CheXbert state dict has a 'model' key in some versions
    sd = state.get("model", state)
    # Keep only bert encoder weights
    bert_sd = {k.replace("bert.", ""): v for k, v in sd.items() if "bert." in k}
    model.load_state_dict(bert_sd, strict=False)
    model.eval()
    return tokenizer, model


def _embed(texts: List[str], tokenizer, model, device: str) -> np.ndarray:
    import torch
    model = model.to(device)
    embeddings = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls)
    return np.array(embeddings)


def compute_semb_score(
    candidates: List[str],
    references: List[str],
    chexbert_path: str,
) -> List[Optional[float]]:
    if not HAS_TORCH:
        warnings.warn("torch not available — skipping SembScore.")
        return [None] * len(candidates)
    if not chexbert_path:
        warnings.warn("chexbert_path not set — skipping SembScore.")
        return [None] * len(candidates)

    try:
        tokenizer, model = _load_chexbert(chexbert_path)
    except Exception as e:
        warnings.warn(f"CheXbert load failed: {e} — skipping SembScore.")
        return [None] * len(candidates)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[name-defined]
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cand_emb = _embed(candidates, tokenizer, model, device)
    ref_emb = _embed(references, tokenizer, model, device)

    # Cosine similarity per pair
    norms_c = np.linalg.norm(cand_emb, axis=1, keepdims=True)
    norms_r = np.linalg.norm(ref_emb, axis=1, keepdims=True)
    cos_sim = (cand_emb * ref_emb).sum(axis=1) / (norms_c.squeeze() * norms_r.squeeze() + 1e-8)
    return cos_sim.tolist()
