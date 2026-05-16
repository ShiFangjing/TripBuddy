import sys
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_self_check_passes():
    cmd = [sys.executable, str(ROOT / "cli_langgraph.py"), "--self-check"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "self-check passed" in (proc.stdout + proc.stderr)
