"""
llm.py — llama-cpp-python wrapper tuned for LFM2.5-1.2B Q4_K_M on an
Intel i5-7300U (2C/4T, 8 GB RAM).

Key decisions
-------------
* Manual ChatML prompt building (the exact LFM2.5 template) instead of
  create_chat_completion, so the format is guaranteed correct regardless of
  the GGUF metadata shipped with community fine-tunes.
* One model instance shared by the chat thread and background memory workers,
  guarded by a lock — a second context would waste ~0.5 GB RAM we don't have.
* Streaming generation for a responsive UI; blocking complete() for the
  background extraction/summarization jobs.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Iterator, List, Tuple

import config


class LLMEngine:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or config.MODEL_PATH
        self._llama = None
        self._lock = threading.Lock()   # single model, serialized access
        self.last_speed: float = 0.0    # tokens/sec of last generation
        self.loaded: bool = False

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. See README.md.") from e
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"Model file not found:\n  {self.model_path}\n"
                "Set MODEL_PATH in config.py or the LFM_MODEL_PATH env var.")
        kwargs = dict(
            model_path=self.model_path,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,     # 4 = all HT threads of i5-7300U
            n_batch=config.N_BATCH,
            seed=config.SEED,
            verbose=False,
            # ---- performance (see config.py) ----
            type_k=getattr(config, "CACHE_TYPE_K", "f16"),   # 8-bit KV cache
            type_v=getattr(config, "CACHE_TYPE_V", "f16"),   #  halves ctx RAM
            f16kvcache=config.F16_KV,
            use_mmap=config.USE_MMAP,       # instant-ish loads, shared pagecache
            use_mlock=config.USE_MLOCK,
            mock_numa=config.MOCK_NUMA,
        )
        try:
            self._llama = Llama(**kwargs)
        except TypeError:
            # Older llama-cpp-python without KV-cache quantization args:
            # drop them and load anyway instead of crashing.
            for k in ("type_k", "type_v", "f16kvcache", "mock_numa"):
                kwargs.pop(k, None)
            self._llama = Llama(**kwargs)
        self.loaded = True

    # --------------------------------------------------------------- prompt

    @staticmethod
    def build_prompt(system_prompt: str,
                     messages: List[Tuple[str, str]]) -> str:
        """LFM2.5 ChatML: BOS + system/user/assistant blocks, assistant open."""
        parts = [config.BOS]
        if system_prompt:
            parts.append(f"{config.IM_START}system\n{system_prompt}{config.IM_END}\n")
        for role, content in messages:
            parts.append(f"{config.IM_START}{role}\n{content}{config.IM_END}\n")
        parts.append(f"{config.IM_START}assistant\n")
        return "".join(parts)

    # ----------------------------------------------------------- generation

    def _params(self, max_tokens: int, temperature: float | None) -> dict:
        return dict(
            max_tokens=max_tokens,
            temperature=config.TEMPERATURE if temperature is None else temperature,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            repeat_penalty=config.REPEAT_PENALTY,
            stop=list(config.STOP_TOKENS),
        )

    def stream(self, system_prompt: str,
               messages: List[Tuple[str, str]],
               max_tokens: int | None = None) -> Iterator[str]:
        """Streaming chat generation; yields text pieces."""
        prompt = self.build_prompt(system_prompt, messages)
        params = self._params(max_tokens or config.MAX_TOKENS, None)
        t0, n = time.time(), 0
        with self._lock:
            for chunk in self._llama(prompt, stream=True, **params):
                piece = chunk["choices"][0]["text"]
                if piece:
                    n += 1
                    yield piece
        dt = time.time() - t0
        self.last_speed = n / dt if dt > 0 else 0.0

    def complete(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.1) -> str:
        """Blocking raw-prompt completion for background memory jobs."""
        with self._lock:
            out = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                repeat_penalty=config.REPEAT_PENALTY,
                stop=list(config.STOP_TOKENS),
            )
        return out["choices"][0]["text"].strip()
