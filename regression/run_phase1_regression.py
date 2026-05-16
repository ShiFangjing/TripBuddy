#!/usr/bin/env python3
"""Run phase-1 golden prompt regression against LangGraph rewrite CLI.

This runner focuses on deterministic structure checks for phase-1:
- command success/failure
- whether expected migrated agents appear in agent_results
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "regression" / "golden_prompts.json"
CLI_PATH = ROOT / "cli_langgraph.py"
REPORT_PATH = ROOT / "regression" / "phase1_report.json"


def _extract_agent_names(output_text: str) -> List[str]:
    marker = "--- agent_results ---"
    idx = output_text.find(marker)
    if idx < 0:
        return []

    payload_text = output_text[idx + len(marker) :].strip()
    try:
        arr = json.loads(payload_text)
    except json.JSONDecodeError:
        return []

    names = []
    for item in arr:
        if isinstance(item, dict) and item.get("agent_name"):
            names.append(str(item["agent_name"]))
    return names


def run() -> int:
    prompts = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    report: List[Dict[str, Any]] = []

    for case in prompts:
        prompt = case["prompt"]
        focus = case.get("focus", [])

        cmd = [sys.executable, str(CLI_PATH), "--query", prompt]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

        names = _extract_agent_names(proc.stdout or "")
        expected_hits = []
        if "event_collection" in focus:
            expected_hits.append("event_collection")
        if "preference" in focus:
            expected_hits.append("preference")

        matched = all(name in names for name in expected_hits) if expected_hits else True

        report.append(
            {
                "id": case["id"],
                "prompt": prompt,
                "focus": focus,
                "returncode": proc.returncode,
                "agent_names": names,
                "expected_hits": expected_hits,
                "matched": matched,
                "stdout_head": (proc.stdout or "")[:500],
                "stderr_head": (proc.stderr or "")[:500],
            }
        )

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [r for r in report if r["returncode"] != 0 or not r["matched"]]
    print(f"wrote: {REPORT_PATH}")
    print(f"total={len(report)} failed={len(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
