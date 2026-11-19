
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production cost guard.

Checks the canonical film project's requested audio services before
production. Free-only is the default. Unknown and paid services are
blocked unless a separate, explicit approval policy permits them.

This module does not make API calls or claim live prices/quotas.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


SERVICE_CATALOG = {
    "gemini": {
        "name": "Gemini TTS",
        "category": "free_quota",
        "requires_quota_confirmation": True,
    },
    "edge": {
        "name": "Edge TTS",
        "category": "free",
    },
    "gtts": {
        "name": "gTTS",
        "category": "free",
    },
    "xai": {
        "name": "xAI / Grok TTS",
        "category": "paid",
    },
    "fal": {
        "name": "FAL audio generation",
        "category": "paid",
    },
    "licensed_library": {
        "name": "Licensed audio library",
        "category": "license_check",
    },
}

FREE_ONLY_ORDER = ["edge", "gtts"]


class CostPolicyError(RuntimeError):
    pass


def read_json(path):
    path = Path(path)

    if not path.is_file():
        raise CostPolicyError(f"Project JSON not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CostPolicyError(
            f"Invalid project JSON: {exc}"
        ) from exc


def normalize_policy(project):
    raw = project.get("cost_policy", {})

    if not isinstance(raw, dict):
        raw = {}

    mode = str(raw.get("mode", "free_only")).strip().lower()

    if mode not in {"free_only", "approval_required", "paid_allowed"}:
        raise CostPolicyError(f"Unsupported cost mode: {mode}")

    budget = raw.get("max_project_budget", 0)

    try:
        budget = float(budget)
    except (TypeError, ValueError):
        raise CostPolicyError("Invalid project budget.")

    if budget < 0:
        raise CostPolicyError("Project budget cannot be negative.")

    return {
        "mode": mode,
        "paid_services_allowed": (
            raw.get("paid_services_allowed") is True
        ),
        "require_approval": True,
        "block_unknown_prices": True,
        "currency": str(raw.get("currency", "USD")).upper(),
        "max_project_budget": budget,
    }


def get_audio(project):
    audio = project.get("audio", {})

    if not isinstance(audio, dict):
        raise CostPolicyError("Project audio settings are invalid.")

    return audio


def requested_services(audio):
    services = []

    order = audio.get("tts_provider_order", ["gemini", "edge", "gtts", "xai"])

    if not isinstance(order, list):
        raise CostPolicyError("tts_provider_order must be a list.")

    for provider in order:
        provider = str(provider).strip().lower()

        if provider and provider not in services:
            services.append(provider)

    for key, service_name in (
        ("music", "music"),
        ("ambience", "ambience"),
        ("sound_effects", "sound_effects"),
    ):
        setting = audio.get(key, {})

        if isinstance(setting, dict) and setting.get("enabled") is True:
            provider = str(
                setting.get("provider", "")
            ).strip().lower()

            if provider in {"fal", "fal-ai"}:
                services.append("fal")
            elif provider in {"licensed_library", "library"}:
                services.append("licensed_library")
            elif provider:
                services.append(provider)
            else:
                services.append("unknown")

    return list(dict.fromkeys(services))


def inspect_service(service, policy, approval, quota_confirmed):
    info = SERVICE_CATALOG.get(service)

    if info is None:
        return {
            "service": service,
            "status": "BLOCKED",
            "reason": "Unknown provider or unknown pricing.",
        }

    category = info["category"]

    if category == "free":
        return {
            "service": info["name"],
            "status": "ALLOWED",
            "reason": "Classified as free; service availability is not guaranteed.",
        }

    if category == "free_quota":
        if not quota_confirmed:
            return {
                "service": info["name"],
                "status": "BLOCKED",
                "reason": (
                    "Free quota/billing status is not confirmed. "
                    "Use another free provider or verify the account separately."
                ),
            }

        return {
            "service": info["name"],
            "status": "ALLOWED",
            "reason": "Free quota explicitly confirmed by workflow setting.",
        }

    if category == "license_check":
        return {
            "service": info["name"],
            "status": "BLOCKED",
            "reason": (
                "Asset license must be verified before use. "
                "No license evidence was supplied to this checker."
            ),
        }

    if category == "paid":
        if policy["mode"] == "free_only":
            return {
                "service": info["name"],
                "status": "BLOCKED",
                "reason": "Paid services are disabled in free-only mode.",
            }

        if not policy["paid_services_allowed"]:
            return {
                "service": info["name"],
                "status": "BLOCKED",
                "reason": "Project policy does not allow paid services.",
            }

        if not approval:
            return {
                "service": info["name"],
                "status": "BLOCKED",
                "reason": "Explicit workflow approval is missing.",
            }

        return {
            "service": info["name"],
            "status": "BLOCKED",
            "reason": (
                "Live price is not verified. Unknown prices are blocked; "
                "no paid API call may start."
            ),
        }

    return {
        "service": info["name"],
        "status": "BLOCKED",
        "reason": "Unsupported cost classification.",
    }


def build_report(project, approval=False, quota_confirmed=False):
    policy = normalize_policy(project)
    audio = get_audio(project)

    services = requested_services(audio)
    results = [
        inspect_service(
            service,
            policy,
            approval,
            quota_confirmed,
        )
        for service in services
    ]

    blocked = [
        item for item in results
        if item["status"] == "BLOCKED"
    ]

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": policy["mode"],
        "currency": policy["currency"],
        "budget_limit": policy["max_project_budget"],
        "paid_services_allowed": policy["paid_services_allowed"],
        "explicit_workflow_approval": bool(approval),
        "estimated_cost": None,
        "actual_cost": None,
        "cost_note": (
            "No live price or billing API was queried. "
            "Costs are not estimated or invented."
        ),
        "services": results,
        "production_allowed": len(blocked) == 0,
        "blocked_count": len(blocked),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=".work/project.json",
    )
    parser.add_argument(
        "--report",
        default="output/cost_report.json",
    )
    args = parser.parse_args()

    approval = (
        os.environ.get("PAID_SERVICES_APPROVED", "").lower()
        == "true"
    )
    quota_confirmed = (
        os.environ.get("GEMINI_TTS_FREE_QUOTA_CONFIRMED", "").lower()
        == "true"
    )

    try:
        project = read_json(args.project)
        report = build_report(
            project,
            approval=approval,
            quota_confirmed=quota_confirmed,
        )

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("COST PREFLIGHT")
        print("Mode:", report["mode"])
        print("Production allowed:", report["production_allowed"])

        for item in report["services"]:
            print(
                f'{item["status"]}: {item["service"]} - '
                f'{item["reason"]}'
            )

        print("Report:", report_path)

        if not report["production_allowed"]:
            raise CostPolicyError(
                "Production blocked by cost policy. "
                "Review the cost report before continuing."
            )

    except Exception as exc:
        print(f"COST GUARD FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
