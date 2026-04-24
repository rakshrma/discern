from __future__ import annotations
import os
import asyncio
import logging
import time
from functools import lru_cache
from typing import Dict, List, Optional
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

TOKEN_LIMIT_PREFIX = "ERROR:TOKEN_LIMIT"


class TokenLimitError(RuntimeError):
    """Raised when the model's output was cut off at the token cap."""


class InputTooLongError(RuntimeError):
    """Raised when the input prompt exceeds the model's context window."""


DEFAULT_DB_TOKEN = "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/reads/config/.databricks.token"
DEFAULT_HF_TOKEN = "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/reads/config/.hftoken"
DEFAULT_DB_BASE_URL = "https://adb-624977420987022.2.azuredatabricks.net/serving-endpoints"

# --------------- cumulative token counters ---------------
_token_usage: Dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}
_token_lock: Optional[asyncio.Lock] = None


def _get_token_lock() -> asyncio.Lock:
    """Lazy-create the async lock inside the running event loop."""
    global _token_lock
    if _token_lock is None:
        _token_lock = asyncio.Lock()
    return _token_lock


def get_token_usage() -> Dict[str, int]:
    """Return a copy of the cumulative token counts."""
    return dict(_token_usage)


def reset_token_usage() -> None:
    """Reset all counters to zero."""
    global _token_lock
    for k in _token_usage:
        _token_usage[k] = 0
    _token_lock = None  # re-create on next use in case event loop changed


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

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


@lru_cache(maxsize=8)
def _async_client_db(token_path: str, base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=_read(token_path), base_url=base_url)


# ---------------------------------------------------------
# vLLM local inference
# ---------------------------------------------------------

@lru_cache(maxsize=4)
def _load_vllm_model(model: str, hf_token: Optional[str] = None):
    """Load a model using vLLM (first call only, cached thereafter)."""
    import torch
    from vllm import LLM

    requested_tp = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    n_gpus = torch.cuda.device_count()
    if requested_tp > n_gpus:
        raise RuntimeError(
            f"[vLLM] VLLM_TENSOR_PARALLEL_SIZE={requested_tp} but only "
            f"{n_gpus} GPU(s) visible on this node. "
            f"Use --nodes=1 --gpus-per-node={requested_tp} in your SLURM job "
            f"to ensure all GPUs are co-located on a single node."
        )

    print(f"[vLLM] Loading model '{model}' on {n_gpus} GPU(s), TP={requested_tp} ...")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    llm = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=requested_tp,
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEM_UTIL", "0.90")),
        max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
    )
    print(f"[vLLM] Model '{model}' loaded successfully.")
    return llm


def _unload_vllm_model():
    """Free GPU memory held by cached vLLM models."""
    import gc
    import torch

    _load_vllm_model.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    print("[vLLM] GPU memory freed.")


_unload_hf_pipeline = _unload_vllm_model


def _is_thinking_model(model: str) -> bool:
    """Models that generate <think>...</think> reasoning tokens before the answer.
    Qwen3/3.5 can have thinking disabled at template level; DeepSeek-R1 always thinks."""
    name = model.lower()
    return "qwen3" in name or "deepseek-r1" in name or "deepseek_r1" in name


def _apply_chat_template_vllm(tokenizer, messages: List[Dict[str, str]], model: str) -> str:
    """Apply chat template. Disables thinking for Qwen3/3.5 so the model returns
    structured output directly instead of a long <think> preamble."""
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model.lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (DeepSeek-R1 always emits them)."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _query_hf(
    messages: List[Dict[str, str]],
    model: str,
    hf_token: Optional[str],
    max_tokens: int,
    temperature: float,
) -> str:
    """Run single chat-completion locally via vLLM."""
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    llm = _load_vllm_model(model, hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        model, token=hf_token, trust_remote_code=True
    )
    prompt = _apply_chat_template_vllm(tokenizer, messages, model)

    # Cap max_tokens to the model's actual context window to prevent engine crash
    # when callers escalate beyond max_model_len during retries.
    model_max_len = llm.llm_engine.model_config.max_model_len
    safe_max_tokens = min(max_tokens, model_max_len)

    sampling_params = SamplingParams(
        max_tokens=safe_max_tokens,
        temperature=temperature if temperature > 0 else 0,
    )

    t0 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - t0
    output = outputs[0].outputs[0]
    text = output.text.strip()
    if _is_thinking_model(model):
        text = _strip_thinking(text)

    try:
        pt = len(outputs[0].prompt_token_ids)
        ct = len(output.token_ids)
        _token_usage["prompt_tokens"] += pt
        _token_usage["completion_tokens"] += ct
        _token_usage["total_tokens"] += pt + ct
        print(f"[vLLM] prompt_tokens={pt}  completion_tokens={ct}  "
              f"total={pt+ct}  time={elapsed:.2f}s  "
              f"tok/s={ct/elapsed:.1f}")
    except Exception as e:
        logger.warning(f"[vLLM] Token counting failed: {e}")

    if output.finish_reason == "length":
        raise TokenLimitError(
            f"Output truncated: completion_tokens={ct}, max_tokens={safe_max_tokens}"
        )

    return text


def _query_hf_batch(
    messages_batch: List[List[Dict[str, str]]],
    model: str,
    hf_token: Optional[str],
    max_tokens: int,
    temperature: float,
) -> List[str]:
    """
    Batch chat-completion locally via vLLM.

    vLLM natively batches multiple prompts in a single .generate() call,
    using continuous batching on the GPU — significantly faster than
    calling one at a time.
    """
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    llm = _load_vllm_model(model, hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        model, token=hf_token, trust_remote_code=True
    )

    prompts = [
        _apply_chat_template_vllm(tokenizer, msgs, model)
        for msgs in messages_batch
    ]

    model_max_len = llm.llm_engine.model_config.max_model_len
    safe_max_tokens = min(max_tokens, model_max_len)

    sampling_params = SamplingParams(
        max_tokens=safe_max_tokens,
        temperature=temperature if temperature > 0 else 0,
    )

    print(f"[vLLM] Batch generating {len(prompts)} prompts ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    results: List[str] = []
    total_pt, total_ct = 0, 0
    for i, output in enumerate(outputs):
        out = output.outputs[0]
        text = out.text.strip()
        if _is_thinking_model(model):
            text = _strip_thinking(text)

        try:
            pt = len(output.prompt_token_ids)
            ct = len(out.token_ids)
            total_pt += pt
            total_ct += ct
            _token_usage["prompt_tokens"] += pt
            _token_usage["completion_tokens"] += ct
            _token_usage["total_tokens"] += pt + ct
            print(f"  [{i+1}/{len(outputs)}] prompt_tokens={pt}  completion_tokens={ct}  total={pt+ct}")
        except Exception as e:
            logger.warning(f"[vLLM] Token counting failed for prompt {i+1}: {e}")
            ct = 0

        # Signal token limit so batch retry logic can escalate max_tokens
        if out.finish_reason == "length":
            text = f"{TOKEN_LIMIT_PREFIX}:ct={ct},max={safe_max_tokens}"

        results.append(text)

    print(f"[vLLM] Batch complete — {len(results)} responses | "
          f"total_prompt={total_pt}  total_completion={total_ct}  "
          f"time={elapsed:.2f}s  tok/s={total_ct/elapsed:.1f}")
    return results


# ---------------------------------------------------------
# Databricks API (async concurrent batch)
# ---------------------------------------------------------

async def _query_db_single_async(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> str:
    """Single async Databricks API call, gated by a semaphore."""
    async with semaphore:
        try:
            t0 = time.time()
            r = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - t0

            pt = r.usage.prompt_tokens or 0 if r.usage else 0
            ct = r.usage.completion_tokens or 0 if r.usage else 0
            tt = r.usage.total_tokens or 0 if r.usage else 0

            if r.usage:
                async with _get_token_lock():
                    _token_usage["prompt_tokens"] += pt
                    _token_usage["completion_tokens"] += ct
                    _token_usage["total_tokens"] += tt

            print(f"  [DB async {idx+1}/{total}] prompt_tokens={pt}  "
                  f"completion_tokens={ct}  total={tt}  "
                  f"time={elapsed:.2f}s  tok/s={ct/elapsed:.1f}" if elapsed > 0 else "")

            if r.choices[0].finish_reason == "length":
                return f"{TOKEN_LIMIT_PREFIX}:ct={ct},max={max_tokens}"

            if "gpt-oss" in model:
                return r.choices[0].message.content[1].get("text")
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"[DB async] Request {idx+1}/{total} failed: {e}")
            return f"ERROR: {e}"


async def _query_db_batch_async(
    messages_batch: List[List[Dict[str, str]]],
    model: str,
    token_path: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
) -> List[str]:
    """Run batch of Databricks API calls concurrently with a concurrency cap."""
    client = _async_client_db(token_path, base_url)
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(messages_batch)

    tasks = [
        _query_db_single_async(
            client, msgs, model, max_tokens, temperature,
            semaphore, i, total,
        )
        for i, msgs in enumerate(messages_batch)
    ]
    return list(await asyncio.gather(*tasks))


def _query_db_batch(
    messages_batch: List[List[Dict[str, str]]],
    model: str,
    token_path: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int = 3,
) -> List[str]:
    """
    Synchronous wrapper for async Databricks batch.

    Handles the event loop correctly whether called from a script
    (no running loop) or from a Jupyter notebook (loop already running).
    """
    coro = _query_db_batch_async(
        messages_batch, model, token_path, base_url,
        max_tokens, temperature, max_concurrent,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside Jupyter or an already-running event loop
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------
# Public API — single call (unchanged interface)
# ---------------------------------------------------------

def query_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    token_path: str = DEFAULT_DB_TOKEN,
    db_base_url: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_token_path: str = DEFAULT_HF_TOKEN,
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> str:
    """Single chat completion — Databricks API or local vLLM."""
    model = model or "databricks-claude-sonnet-4-5"
    db_base_url = db_base_url or os.getenv("DATABRICKS_SERVING_ENDPOINTS_URL") or DEFAULT_DB_BASE_URL

    if _is_db(model):
        client = _client_db(token_path, db_base_url)
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as api_exc:
            msg = str(api_exc)
            if "input tokens" in msg and ("context length" in msg or "maximum input length" in msg):
                raise InputTooLongError(msg) from api_exc
            raise
        elapsed = time.time() - t0

        pt = r.usage.prompt_tokens or 0 if r.usage else 0
        ct = r.usage.completion_tokens or 0 if r.usage else 0
        tt = r.usage.total_tokens or 0 if r.usage else 0

        if r.usage:
            _token_usage["prompt_tokens"] += pt
            _token_usage["completion_tokens"] += ct
            _token_usage["total_tokens"] += tt

        print(f"[DB] prompt_tokens={pt}  completion_tokens={ct}  "
              f"total={tt}  time={elapsed:.2f}s  "
              f"tok/s={ct/elapsed:.1f}" if elapsed > 0 else "")

        if r.choices[0].finish_reason == "length":
            raise TokenLimitError(
                f"Output truncated: completion_tokens={ct}, max_tokens={max_tokens}"
            )

        if "gpt-oss" in model:
            return r.choices[0].message.content[1].get("text")
        return (r.choices[0].message.content or "").strip()

    else:
        token = _read(hf_token_path)
        return _query_hf(messages, model, token, max_tokens, temperature)


# ---------------------------------------------------------
# Public API — batch call
# ---------------------------------------------------------

def query_llm_batch(
    messages_batch: List[List[Dict[str, str]]],
    model: Optional[str] = None,
    token_path: str = DEFAULT_DB_TOKEN,
    db_base_url: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_token_path: str = DEFAULT_HF_TOKEN,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    max_concurrent: int = 3,
) -> List[str]:
    """
    Batch chat completion — sends multiple message lists at once.

    For Databricks API: fires concurrent async requests (capped by
    max_concurrent to respect rate limits).

    For local vLLM: passes all prompts in a single .generate() call
    for GPU-efficient continuous batching.

    Parameters
    ----------
    messages_batch : list of list of dict
        Each element is a full messages list (system + user) for one request.
    model : str, optional
        Model name. Defaults to databricks-claude-sonnet-4-5.
    token_path : str
        Databricks token file path.
    db_base_url : str, optional
        Databricks serving endpoint URL.
    hf_token : str, optional
        HuggingFace token string (takes priority over hf_token_path).
    hf_token_path : str
        Path to HuggingFace token file.
    max_tokens : int
        Max tokens per completion (default 2000).
    temperature : float
        Sampling temperature (default 0.1).
    max_concurrent : int
        Max concurrent API requests for Databricks (default 5).
        Ignored for vLLM (all prompts batched in one .generate() call).

    Returns
    -------
    list[str]
        Responses in the same order as messages_batch.
    """
    model = model or "databricks-claude-sonnet-4-5"
    db_base_url = db_base_url or os.getenv("DATABRICKS_SERVING_ENDPOINTS_URL") or DEFAULT_DB_BASE_URL

    if not messages_batch:
        return []

    backend = "Databricks API" if _is_db(model) else "vLLM local"
    print(f"[Batch] {len(messages_batch)} requests via {backend} (model={model})")

    if _is_db(model):
        return _query_db_batch(
            messages_batch=messages_batch,
            model=model,
            token_path=token_path,
            base_url=db_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
        )
    else:
        token = hf_token or _read(hf_token_path)
        return _query_hf_batch(
            messages_batch=messages_batch,
            model=model,
            hf_token=token,
            max_tokens=max_tokens,
            temperature=temperature,
        )