#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cihan AI Film Studio - central, free-first production controller.

This module orchestrates existing project, cost, checkpoint, audio and
MuseTalk components. It deliberately does not pretend that a remote Kaggle
GPU job, a third-party model quota, or social publishing has succeeded unless
an actual artifact/result is available.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

VERSION = "2.0.0"
DEFAULT_WORK = Path(".work/film_studio")
DEFAULT_PROJECT = DEFAULT_WORK / "project.json"
DEFAULT_REPORTS = DEFAULT_WORK / "reports"
DEFAULT_JOBS = DEFAULT_WORK / "jobs"

# These are the checkpoint stages already defined by production_orchestrator.py.
STAGES = (
    "project_validation", "script", "character_assets", "video_generation",
    "audio_generation", "post_production", "quality_control", "publishing",
)


class StudioError(RuntimeError):
    """A safe, user-facing production error."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, level: str = "INFO") -> None:
    print(f"[FILM-STUDIO][{level}] {message}", flush=True)


def json_read(path: str | Path) -> Any:
    target = Path(path)
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudioError(f"Dosya bulunamadÄ±: {target}") from exc
    except json.JSONDecodeError as exc:
        raise StudioError(f"GeÃ§ersiz JSON ({target}): {exc}") from exc


def json_write(path: str | Path, data: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def free_policy(project: dict[str, Any], gemini_quota_confirmed: bool = False) -> dict[str, Any]:
    """Enforce the repo's free-only policy without inventing service quotas."""
    audio = project.setdefault("audio", {})
    if not isinstance(audio, dict):
        raise StudioError("Projedeki audio alanÄ± nesne olmalÄ±.")

    # Edge TTS and gTTS are the only default speech routes in free-only mode.
    # Gemini is added only after an explicit quota confirmation and API key.
    order = ["edge", "gtts"]
    if gemini_quota_confirmed and os.getenv("GEMINI_API_KEY", "").strip():
        order.insert(0, "gemini")
    audio["tts_provider_order"] = order

    # Do not silently activate generated audio services. Licensed local files
    # can be enabled by the project only if the cost guard validates them.
    for key in ("music", "ambience", "sound_effects"):
        setting = audio.get(key)
        if isinstance(setting, dict):
            provider = str(setting.get("provider", "")).strip().lower()
            if provider in {"fal", "fal-ai", "xai", "paid"}:
                setting["enabled"] = False
                setting["provider"] = "licensed_library"
            # Disabled by default until there is a verified asset or tested
            # free generation adapter; the audio mixer still supports assets.
            if not setting.get("asset_path"):
                setting["enabled"] = False

    project["cost_policy"] = {
        "mode": "free_only", "paid_services_allowed": False,
        "require_approval": True, "show_cost_before_run": True,
        "block_unknown_prices": True, "currency": "USD",
        "max_project_budget": 0.0, "record_estimated_costs": True,
        "record_actual_costs": True,
    }
    return project


def validate_project(project: dict[str, Any]) -> dict[str, Any]:
    from film_project import validate_film_project
    report = validate_film_project(project)
    if not report.get("valid"):
        details = "\n".join(f"- {item}" for item in report.get("errors", []))
        raise StudioError("Film projesi doÄrulanamadÄ±:\n" + details)
    return report


def configure_free_llm_routes() -> list[str]:
    """Allow only the explicit OpenRouter free router or user-confirmed free quotas."""
    order: list[str] = []
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        # Pin to the explicitly free router; never inherit a paid model override.
        os.environ["OPENROUTER_MODEL"] = "openrouter/free"
        order.append("openrouter")
    if (
        os.getenv("GROQ_API_KEY", "").strip()
        and os.getenv("GROQ_FREE_QUOTA_CONFIRMED", "false").lower() == "true"
    ):
        order.append("groq")
    if (
        os.getenv("GEMINI_API_KEY", "").strip()
        and os.getenv("GEMINI_LLM_FREE_QUOTA_CONFIRMED", "false").lower() == "true"
    ):
        order.append("gemini")
    os.environ["LLM_PROVIDER_ORDER"] = ",".join(order)
    if not order:
        raise StudioError(
            "Ãcretsiz senaryo saÄlayÄ±cÄ±sÄ± yapÄ±landÄ±rÄ±lmamÄ±Å. "
            "OPENROUTER_API_KEY (openrouter/free) ekleyin veya Groq/Gemini "
            "Ã¼cretsiz kotasÄ±nÄ± ayrÄ±ca doÄrulayÄ±p ilgili *_FREE_QUOTA_CONFIRMED "
            "ayarÄ±nÄ± true yapÄ±n. Ãcretli modele otomatik geÃ§ilmeyecek."
        )
    return order


def build_project(args: argparse.Namespace) -> int:
    if not args.topic.strip():
        raise StudioError("Film konusu boÅ olamaz.")
    if not 15 <= args.duration <= 180:
        raise StudioError("Film sÃ¼resi 15-180 saniye arasÄ±nda olmalÄ±.")

    os.environ["VIDEO_LANGUAGE"] = args.language
    os.environ["TARGET_SECONDS"] = str(args.duration)
    os.environ["GEMINI_TTS_FREE_QUOTA_CONFIRMED"] = str(bool(args.gemini_quota_confirmed)).lower()
    llm_routes = configure_free_llm_routes()
    from topic_project_generator import generate_project
    from production_cost_guard import build_report

    log("Senaryo, karakterler, sahneler ve diyaloglar oluÅturuluyor.")
    project = generate_project(args.topic, args.style)
    if not isinstance(project, dict):
        raise StudioError("Senaryo Ã¼reticisi beklenen proje nesnesini dÃ¶ndÃ¼rmedi.")

    free_policy(project, args.gemini_quota_confirmed)
    validation = validate_project(project)
    output = Path(args.output)
    cost_path = Path(args.cost_report)
    cost_report = build_report(
        project,
        approval=False,
        quota_confirmed=args.gemini_quota_confirmed,
        project_dir=output.resolve().parent,
    )
    json_write(cost_path, cost_report)
    if cost_report.get("production_allowed") is not True:
        blocked = [
            f"{x.get('service')}: {x.get('reason')}"
            for x in cost_report.get("services", [])
            if x.get("status") == "BLOCKED"
        ]
        raise StudioError("Maliyet kontrolÃ¼ Ã¼retimi durdurdu. " + " | ".join(blocked))

    json_write(output, project)
    job_id = str(uuid.uuid4())
    job_dir = Path(args.job_root) / job_id
    from production_orchestrator import ProductionJob
    job = ProductionJob.create(job_dir, project)

    report = {
        "studio_version": VERSION,
        "status": "prepared",
        "created_at": now_iso(),
        "project_file": str(output),
        "project_sha256": hashlib.sha256(json.dumps(project, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "cost_report": str(cost_path),
        "job_workspace": str(job.workspace),
        "job_manifest": str(job.manifest_path),
        "project_title": project.get("film", {}).get("title", ""),
        "scene_count": len(project.get("scenes", [])),
        "character_count": len(project.get("characters", [])),
        "dialogue_count": validation.get("dialogue_turn_count", 0),
        "free_only": True,
        "llm_routes_allowed": llm_routes,
        "paid_services_allowed": False,
        "not_claimed_as_complete": [
            "Kaggle video generation is a separate remote workflow stage.",
            "MuseTalk requires a real GPU worker result before compositing.",
            "Music/SFX generation and social publishing are not enabled by this controller.",
        ],
    }
    json_write(Path(args.status), report)
    log(f"Proje hazÄ±r: {output}")
    log(f"BaÅlÄ±k: {report['project_title']}")
    log(f"Sahne: {report['scene_count']} | karakter: {report['character_count']} | diyalog: {report['dialogue_count']}")
    log(f"Ä°Å takip dosyasÄ±: {job.manifest_path}")
    log("Ãcretsiz senaryo rotalarÄ±: " + ", ".join(llm_routes))
    print("FILM_STUDIO_STATUS=prepared")
    print(f"PROJECT_OUTPUT={output.resolve()}")
    print(f"JOB_WORKSPACE={job.workspace}")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    project = json_read(args.project)
    report = validate_project(project)
    log("Proje ÅemasÄ± geÃ§erli.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def normalize_wan_command(args: argparse.Namespace) -> int:
    project = json_read(args.project)
    from wan_project_input import normalize_wan_project
    normalized = normalize_wan_project(project)
    json_write(args.output, normalized)
    log(f"Wan uyumlu proje oluÅturuldu: {args.output}")
    print(f"WAN_PROJECT_OUTPUT={Path(args.output).resolve()}")
    return 0


def cost_command(args: argparse.Namespace) -> int:
    project = json_read(args.project)
    from production_cost_guard import build_report
    report = build_report(
        project, approval=False,
        quota_confirmed=args.gemini_quota_confirmed,
        project_dir=Path(args.project).resolve().parent,
    )
    json_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("production_allowed") else 2


def load_job(workspace: str):
    from production_orchestrator import ProductionJob
    return ProductionJob.open(workspace)


def job_status(args: argparse.Namespace) -> int:
    job = load_job(args.workspace)
    print(json.dumps(job.manifest, ensure_ascii=False, indent=2))
    problems = job.verify_artifacts()
    if problems:
        log("Artifact doÄrulama hatasÄ±: " + " | ".join(problems), "ERROR")
        return 2
    return 0


def job_stage(args: argparse.Namespace) -> int:
    job = load_job(args.workspace)
    if args.action == "start":
        attempt = job.start_stage(args.stage)
        log(f"AÅama baÅladÄ±: {args.stage} (deneme {attempt})")
    elif args.action == "complete":
        job.complete_stage(args.stage, args.artifact or [])
        log(f"AÅama tamamlandÄ± ve dosyalar hash ile doÄrulandÄ±: {args.stage}")
    elif args.action == "fail":
        job.fail_stage(args.stage, args.reason)
        log(f"AÅama baÅarÄ±sÄ±z olarak kaydedildi: {args.stage}", "ERROR")
    elif args.action == "retry":
        job.retry_failed_stage(args.stage)
        log(f"BaÅarÄ±sÄ±z aÅama kontrollÃ¼ yeniden denemeye hazÄ±r: {args.stage}")
    print(json.dumps(job.manifest, ensure_ascii=False, indent=2))
    return 0


def doctor(args: argparse.Namespace) -> int:
    modules = [
        "film_project", "pipeline_config", "project_adapter",
        "topic_project_generator", "wan_project_input",
        "wan_kaggle_workflow_prepare", "wan_audio_postprocess",
        "production_cost_guard", "production_orchestrator",
        "character_asset_generator", "musetalk_lipsync_prepare",
        "musetalk_kaggle_worker", "musetalk_film_compositor",
        "audio_asset_mixer", "quality_gate",
    ]
    commands = ("python", "ffmpeg", "ffprobe", "kaggle", "git")
    secrets = (
        "OPENROUTER_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
        "GEMINI_TTS_FREE_QUOTA_CONFIRMED", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN",
    )
    report = {
        "studio_version": VERSION, "checked_at": now_iso(),
        "modules": {name: module_available(name) for name in modules},
        "commands": {name: shutil.which(name) for name in commands},
        "environment_variables_present": {name: bool(os.getenv(name, "").strip()) for name in secrets},
        "secret_values_exposed": False,
        "policy": "free_only; no paid automatic fallback",
        "warnings": [
            "API key presence does not prove free quota or service availability.",
            "Kaggle GPU access depends on account quota, queue and session limits.",
            "A successful preflight is not proof that a complete film was rendered.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [name for name, available in report["modules"].items() if not available]
    return 1 if missing else 0


def ffprobe(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise StudioError(f"Medya dosyasÄ± yok veya boÅ: {target}")
    binary = shutil.which("ffprobe")
    if not binary:
        raise StudioError("ffprobe bulunamadÄ±; FFmpeg kurulumu gerekli.")
    proc = subprocess.run(
        [binary, "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height", "-of", "json", str(target)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode:
        raise StudioError(f"ffprobe baÅarÄ±sÄ±z ({target}): {proc.stderr[-1500:]}")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    duration = float((data.get("format") or {}).get("duration") or 0)
    types = [x.get("codec_type") for x in streams]
    return {
        "path": str(target.resolve()), "size_bytes": target.stat().st_size,
        "duration_seconds": round(duration, 3), "stream_types": types,
        "has_video": "video" in types, "has_audio": "audio" in types,
        "streams": streams,
    }


def validate_srt(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        raise StudioError(f"AltyazÄ± dosyasÄ± yok veya boÅ: {target}")
    raw = target.read_text(encoding="utf-8-sig", errors="replace")
    import re
    blocks = [b for b in re.split(r"\r?\n\s*\r?\n", raw.strip()) if b.strip()]
    valid = 0
    timestamp = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3 and timestamp.fullmatch(lines[1].strip()) and any(x.strip() for x in lines[2:]):
            valid += 1
    if valid != len(blocks) or not valid:
        raise StudioError(f"SRT doÄrulamasÄ± baÅarÄ±sÄ±z: {valid}/{len(blocks)} geÃ§erli blok.")
    return {"path": str(target.resolve()), "cue_count": valid, "size_bytes": target.stat().st_size}


def quality_control(args: argparse.Namespace) -> int:
    video = ffprobe(args.video)
    problems: list[str] = []
    if not video["has_video"]:
        problems.append("Video akÄ±ÅÄ± bulunamadÄ±.")
    if args.require_audio and not video["has_audio"]:
        problems.append("Final videoda ses akÄ±ÅÄ± bulunamadÄ±.")
    if video["duration_seconds"] <= 0:
        problems.append("Video sÃ¼resi okunamadÄ±.")
    subtitle = validate_srt(args.subtitles) if args.subtitles else None
    audio = ffprobe(args.audio) if args.audio else None
    if audio and not audio["has_audio"]:
        problems.append("AyrÄ± ses dosyasÄ±nda ses akÄ±ÅÄ± yok.")
    if args.expected_duration and abs(video["duration_seconds"] - args.expected_duration) > args.duration_tolerance:
        problems.append(f"SÃ¼re hedefle uyuÅmuyor: hedef {args.expected_duration}s, Ã§Ä±ktÄ± {video['duration_seconds']}s.")
    report = {
        "studio_version": VERSION, "checked_at": now_iso(),
        "status": "failed" if problems else "passed",
        "video": video, "audio": audio, "subtitles": subtitle,
        "problems": problems,
        "limitations": [
            "Bu kontrol medya akÄ±ÅÄ±nÄ± ve SRT yapÄ±sÄ±nÄ± inceler; yaratÄ±cÄ± kaliteyi Ã¶lÃ§mez.",
            "Dudak senkronunun gÃ¶rsel doÄruluÄu ayrÄ±ca insan veya Ã¶zel model deÄerlendirmesi gerektirir.",
            "Karakter sÃ¼rekliliÄi, renk, kurgu ve diyalog anlamÄ± otomatik olarak onaylanmÄ±Å sayÄ±lmaz.",
        ],
    }
    json_write(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if problems else 0


def prepare_lipsync(args: argparse.Namespace) -> int:
    from musetalk_lipsync_prepare import prepare_package
    result = prepare_package(Path(args.video), Path(args.timeline), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def compose_lipsync(args: argparse.Namespace) -> int:
    from musetalk_film_compositor import compose
    result = compose(Path(args.video), Path(args.report), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="film_studio.py",
        description="Cihan AI Film Studio - central free-first production controller",
    )
    commands = root.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prepare", help="Sadece konu vererek doÄrulanmÄ±Å film projesi oluÅtur.")
    p.add_argument("--topic", required=True)
    p.add_argument("--style", default="cinematic realistic")
    p.add_argument("--language", default="Turkish")
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--output", default=str(DEFAULT_PROJECT))
    p.add_argument("--cost-report", default=str(DEFAULT_REPORTS / "cost_report.json"))
    p.add_argument("--status", default=str(DEFAULT_REPORTS / "studio_status.json"))
    p.add_argument("--job-root", default=str(DEFAULT_JOBS))
    p.add_argument("--gemini-quota-confirmed", action="store_true")
    p.set_defaults(func=build_project)

    p = commands.add_parser("validate", help="Canonical film project schema validation.")
    p.add_argument("--project", required=True)
    p.set_defaults(func=validate_command)

    p = commands.add_parser("normalize-wan", help="Convert canonical project to the existing Wan worker schema.")
    p.add_argument("--project", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=normalize_wan_command)

    p = commands.add_parser("cost", help="Re-run the existing free-only cost guard.")
    p.add_argument("--project", required=True)
    p.add_argument("--output", default=str(DEFAULT_REPORTS / "cost_report.json"))
    p.add_argument("--gemini-quota-confirmed", action="store_true")
    p.set_defaults(func=cost_command)

    p = commands.add_parser("doctor", help="Check modules, binaries and secret presence without revealing values.")
    p.set_defaults(func=doctor)

    p = commands.add_parser("status", help="Read checkpoint manifest and verify artifact hashes.")
    p.add_argument("--workspace", required=True)
    p.set_defaults(func=job_status)

    p = commands.add_parser("stage", help="Manage a real checkpoint; completion requires existing artifacts.")
    p.add_argument("action", choices=("start", "complete", "fail", "retry"))
    p.add_argument("--workspace", required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--artifact", action="append", help="Artifact path relative to job workspace; repeatable.")
    p.add_argument("--reason", default="Stage failed; reason not supplied.")
    p.set_defaults(func=job_stage)

    p = commands.add_parser("qc", help="Validate final video/audio streams and subtitle structure.")
    p.add_argument("--video", required=True)
    p.add_argument("--audio")
    p.add_argument("--subtitles")
    p.add_argument("--require-audio", action="store_true")
    p.add_argument("--expected-duration", type=float)
    p.add_argument("--duration-tolerance", type=float, default=2.0)
    p.add_argument("--report", default=str(DEFAULT_REPORTS / "final_qc.json"))
    p.set_defaults(func=quality_control)

    p = commands.add_parser("prepare-lipsync", help="Build the verified MuseTalk input package from the existing timeline.")
    p.add_argument("--video", required=True)
    p.add_argument("--timeline", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=prepare_lipsync)

    p = commands.add_parser("compose-lipsync", help="Composite actual MuseTalk worker outputs onto the film.")
    p.add_argument("--video", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=compose_lipsync)
    return root


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log("Ä°Ålem kullanÄ±cÄ± tarafÄ±ndan durduruldu.", "WARNING")
        return 130
    except Exception as exc:
        log(f"{type(exc).__name__}: {exc}", "ERROR")
        if os.getenv("FILM_STUDIO_DEBUG", "").lower() == "true":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
