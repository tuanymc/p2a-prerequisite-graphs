# -*- coding: utf-8 -*-
"""C1: score Qwen 2.5 7B on ESCO split E test (B1 then B3 then B5). Resume-safe."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CFG = "configs/qwen_openrouter.yaml"


def run(cond: str) -> None:
    cmd = [
        PY,
        "-m",
        "src.run_pipeline",
        "--config",
        CFG,
        "--phase",
        "1",
        "--condition",
        cond,
        "--only",
        "--scheme",
        "E",
        "--parts",
        "test",
        "--workers",
        "8",
    ]
    print("start", cond, flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f"{cond} failed rc={rc}")
    print("done", cond, flush=True)


def main() -> None:
    for cond in ("B1", "B3", "B5"):
        run(cond)


if __name__ == "__main__":
    main()
