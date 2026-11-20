"""Print what is still needed before the arena is production ready.

    sudo -u ampixa-arena bash -c 'set -a; . /etc/ampixa-arena/environment; \
        cd /opt/ampixa-arena/current && .venv/bin/python -m submission_app.doctor --data /var/lib/ampixa-arena'

Exit status is 0 when nothing blocks launch, 1 otherwise. `--json` prints the raw report.
"""

import argparse
import json
import os
from pathlib import Path

from .readiness import checks, summary
from .storage import initialize

MARK = {"ok": "✓", "todo": "✗", "warn": "!", "info": "·"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("NEPTTS_DATA", ".local/submissions")))
    parser.add_argument("--online", action="store_true", help="Also query Hugging Face for repo privacy.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    initialize(args.data)
    report = summary(checks(args.data, online=args.online))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        feature = None
        for r in sorted(report["checks"], key=lambda r: r["feature"]):
            if r["feature"] != feature:
                feature = r["feature"]
                state = "ready" if report["features"][feature] else "NOT READY"
                print(f"\n{feature.upper()} — {state}")
            print(f"  {MARK[r['status']]} {r['title']}: {r['detail']}")
            if r["status"] in ("todo", "warn"):
                print(f"      → {r['fix']}")
        print("\nReady to launch." if report["ready"] else "\nSome required steps remain (✗).")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
