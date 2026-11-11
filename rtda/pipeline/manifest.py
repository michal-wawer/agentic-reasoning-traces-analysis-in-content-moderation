"""Run manifests: everything needed to audit or reproduce one pipeline invocation."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from rtda.config import REPO_ROOT, ExperimentConfig

_PACKAGES = [
    "rtda", "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "sentence-transformers",
    "transformers", "torch", "datasets", "openai", "pydantic",
]  # fmt: skip


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> dict[str, str | None]:
    out = {}
    for pkg in _PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def write_manifest(
    run_dir: Path, command: str, cfg: ExperimentConfig, extra: dict[str, Any]
) -> Path:
    manifest = {
        "command": command,
        "argv": sys.argv,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "config_sha256": cfg.fingerprint(),
        "config": cfg.model_dump(mode="json"),
        **extra,
    }
    out_dir = Path(run_dir) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_{command}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
