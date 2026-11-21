#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the existing Cihan Film Studio with a local Qwen3 GGUF model."""
from __future__ import annotations

import os
import sys
from functools import lru_cache


def _load_model():
    model_path = os.environ.get("LOCAL_QWEN_MODEL_PATH", "").strip()
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError(
            "LOCAL_QWEN_MODEL_PATH geçerli bir GGUF dosyasını göstermiyor."
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python kurulu değil. Yerel model çalışma zamanı yüklenemedi."
        ) from exc

    return Llama(
        model_path=model_path,
        n_ctx=int(os.environ.get("LOCAL_QWEN_CONTEXT", "8192")),
        n_threads=max(2, min(os.cpu_count() or 4, 6)),
        n_batch=256,
        n_gpu_layers=0,
        verbose=False,
    )


@lru_cache(maxsize=1)
def _model():
    return _load_model()


def local_qwen_generate(prompt: str) -> str:
    """Return one JSON response from the local model; no remote API is used."""
    model = _model()
    response = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional Turkish-language screenwriter and film director. "
                    "Return exactly one complete valid JSON object. No Markdown. "
                    "Follow the user's schema exactly. /no_think"
                ),
            },
            {
                "role": "user",
                "content": prompt + "\n\n/no_think\nJSON only. Complete every required scene.",
            },
        ],
        temperature=0.55,
        top_p=0.9,
        max_tokens=int(os.environ.get("LOCAL_QWEN_MAX_TOKENS", "12000")),
        response_format={"type": "json_object"},
    )
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Yerel Qwen modelinden yanıt alınamadı.")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Yerel Qwen boş yanıt döndürdü.")
    return content.strip()


def main() -> int:
    import topic_project_generator as generator
    import film_studio

    # Reuse the existing prompt builder, project normalizer, validator,
    # checkpoint system and cost guard. Only the text-generation provider
    # is replaced with local inference.
    generator.PROVIDERS["local_qwen"] = local_qwen_generate
    generator.provider_order = lambda: ["local_qwen"]
    film_studio.configure_free_llm_routes = lambda: ["local_qwen"]

    os.environ["LLM_PROVIDER_ORDER"] = "local_qwen"
    return film_studio.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"LOCAL_FILM_STUDIO_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
        raise
