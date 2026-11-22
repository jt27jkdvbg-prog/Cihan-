
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the existing Cihan Film Studio with a local Qwen3 GGUF model."""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from typing import Any


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
    """Generate one JSON object using only the local GGUF model."""
    model = _model()

    response = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional screenwriter and film director. "
                    "Return exactly one complete JSON object. No Markdown. "
                    "The top-level keys characters and scenes MUST each be JSON arrays. "
                    "characters must look like [{\"id\":\"character_01\",\"name\":\"...\"}]. "
                    "scenes must look like [{\"scene_id\":\"scene_01\"}]. "
                    "Never return either array as an object, string, or null. /no_think"
                ),
            },
            {
                "role": "user",
                "content": (
                    prompt
                    + "\n\nFINAL JSON CHECK: top-level characters is an ARRAY [ ... ]; "
                    + "top-level scenes is an ARRAY [ ... ]. Not objects { ... }. "
                    + "Return JSON only. Complete every required scene. /no_think"
                ),
            },
        ],
        temperature=0.35,
        top_p=0.85,
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


def _decode_json_container(value: Any, field: str) -> Any:
    """Decode JSON accidentally returned as a string."""
    if isinstance(value, str):
        candidate = value.strip()

        if candidate.startswith(("[", "{")):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    return value


def _as_object_list(value: Any, field: str) -> list:
    """Convert common keyed-object/list wrappers into the schema's list form."""
    value = _decode_json_container(value, field)

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        for wrapper_key in (field, "items", "data", "list", "results"):
            nested = value.get(wrapper_key)

            if isinstance(nested, list):
                return nested

        values = list(value.values())

        if values and all(isinstance(item, dict) for item in values):
            return values

    raise ValueError(
        f"{field} alanı listeye dönüştürülemedi; "
        f"gelen tür: {type(value).__name__}."
    )


def _normalize_script_shape(script: Any) -> dict:
    """Unwrap common model wrappers and normalize characters/scenes."""
    if not isinstance(script, dict):
        raise ValueError("Modelin ürettiği ana JSON değeri nesne olmalı.")

    source = dict(script)

    for wrapper_key in ("film_project", "project", "film", "output"):
        nested = source.get(wrapper_key)

        if isinstance(nested, dict) and (
            "characters" in nested or "scenes" in nested
        ):
            merged = dict(source)
            merged.update(nested)
            source = merged
            break

    for field in ("characters", "scenes"):
        if field not in source:
            raise ValueError(
                f"Model çıktısında zorunlu '{field}' alanı yok."
            )

        source[field] = _as_object_list(source[field], field)

    return source


def main() -> int:
    import topic_project_generator as generator
    import film_studio

    # Keep the existing canonical validator, normalizer,
    # cost guard, checkpoint system and CLI.
    original_builder = generator.build_film_project

    def compatible_builder(script, *args, **kwargs):
        normalized = _normalize_script_shape(script)

        print(
            "[LOCAL-QWEN] JSON shape normalized: "
            f"characters={len(normalized['characters'])}, "
            f"scenes={len(normalized['scenes'])}",
            flush=True,
        )

        return original_builder(normalized, *args, **kwargs)

    generator.build_film_project = compatible_builder
    generator.PROVIDERS["local_qwen"] = local_qwen_generate
    generator.provider_order = lambda: ["local_qwen"]

    film_studio.configure_free_llm_routes = lambda: ["local_qwen"]

    os.environ["LLM_PROVIDER_ORDER"] = "local_qwen"

    return film_studio.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"LOCAL_FILM_STUDIO_ERROR={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

