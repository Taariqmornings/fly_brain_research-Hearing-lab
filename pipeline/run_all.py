"""End-to-end pipeline: data -> subnet -> sim gate -> experiments -> export. Usage: run_all.py [--quick|--smoke]."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [("1 DATA", ["data.py"]), ("2 SUBNET", ["subnet.py"]), ("4 SIM", ["sim.py"]),
         ("5 EXPERIMENTS", ["experiments.py"]), ("6 EXPORT", ["export.py"])]


def main() -> None:
    extra = [a for a in sys.argv[1:] if a in ("--quick", "--smoke")]
    log_dir = HERE.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, cmd in STEPS:
        args = cmd + (extra if cmd[0] == "experiments.py" else [])
        log = log_dir / f"{cmd[0].removesuffix('.py')}.log"
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, *args], cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        lines = log.read_text(errors="replace").strip().splitlines()
        if rc:
            print(f"PHASE {name}: FAIL - see {log}")
            print("\n".join(lines[-20:]))
            sys.exit(rc)
        print(f"PHASE {name}: PASS ({time.time() - t0:.0f}s) - {lines[-1] if lines else ''}")


if __name__ == "__main__":
    main()
