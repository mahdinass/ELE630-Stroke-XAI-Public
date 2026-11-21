#!/usr/bin/env python3
"""Local aggregate-only reporting; accepts the sanitized Stage7 results JSON."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        from ele630.xai_reporting import write_report
        write_report(json.loads(args.aggregate.read_text(encoding="utf-8-sig")), args.output, figures=args.figures)
        print("STAGE7_AGGREGATE_REPORT_WRITTEN")
        return 0
    except Exception:
        print("STAGE7_REPORT_FAIL: invalid aggregate, dependency or exclusive output check", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
