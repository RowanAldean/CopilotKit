"""entrypoint.sh refuses to start without OPENAI_API_KEY.

Every model call goes through the OpenAI shim, which cannot start without the
key, so the agent would die at import. The guard must exit before launching
any process; with the key set the script must get past it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ENTRYPOINT = Path(__file__).resolve().parents[2] / "entrypoint.sh"


def _run(env, tmp_path):
    # Stub python / npx so a run that gets past the guard exits at once
    # instead of launching uvicorn and Next.
    for name in ("python", "npx", "node"):
        stub = tmp_path / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    full_env = {
        "PATH": f"{tmp_path}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        **env,
    }
    return subprocess.run(
        ["bash", str(_ENTRYPOINT)],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )


def test_missing_key_is_fatal_before_anything_starts(tmp_path):
    result = _run({}, tmp_path)
    assert result.returncode == 1
    assert "FATAL: OPENAI_API_KEY not set" in result.stderr
    assert "Starting Python agent" not in result.stdout


def test_a_key_gets_past_the_guard(tmp_path):
    result = _run({"OPENAI_API_KEY": "sk-test"}, tmp_path)
    assert "FATAL: OPENAI_API_KEY" not in result.stderr
    assert "Starting Python agent" in result.stdout
