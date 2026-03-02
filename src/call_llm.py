from __future__ import annotations
import os
from functools import lru_cache
from typing import Dict, List, Optional
from openai import OpenAI


DEFAULT_DB_TOKEN = "../config/.databricks.token"
DEFAULT_HF_TOKEN = "../config/.hftoken"
DEFAULT_DB_BASE_URL = "https://adb-624977420987022.2.azuredatabricks.net/serving-endpoints"


def _is_db(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("databricks-") or ("databricks" in m)


@lru_cache(maxsize=8)
def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read().strip()
    if not t:
        raise RuntimeError(f"Empty token file: {path}")
    return t


@lru_cache(maxsize=8)
def _client_db(token_path: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=_read(token_path), base_url=base_url)


@lru_cache(maxsize=4)
def _load_hf_pipeline(model: str, hf_token: Optional[str] = None):
    import torch
    from transformers import pipeline
    
    print(f"[HF] Loading model '{model}' locally... (first call only)")
    return pipeline(
        task="text-generation",
        model=model,
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.bfloat16
    )

def _unload_hf_pipeline():
    import torch, gc
    _load_hf_pipeline.cache_clear()  # clear lru_cache
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[HF] GPU memory freed.")


def _query_hf(
    messages: List[Dict[str, str]],
    model: str,
    hf_token: Optional[str],
    max_tokens: int,
    temperature: float,
) -> str:
    pipe = _load_hf_pipeline(model, hf_token)
    output = pipe(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()


def query_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    token_path: str = DEFAULT_DB_TOKEN,
    db_base_url: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_token_path: str = DEFAULT_HF_TOKEN,       # fix 3: expose so _read() is used
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> str:
    model = model or "databricks-claude-sonnet-4-5"
    db_base_url = db_base_url or os.getenv("DATABRICKS_SERVING_ENDPOINTS_URL") or DEFAULT_DB_BASE_URL

    if _is_db(model):
        client = _client_db(token_path, db_base_url)
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if "gpt-oss" in model:
            return r.choices[0].message.content[1].get("text")
        return (r.choices[0].message.content or "").strip()

    else:
        token = _read(hf_token_path)
        return _query_hf(messages, model, token, max_tokens, temperature)