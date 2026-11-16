import argparse
import json
import sys
from pathlib import Path

from .lab import publish, train
from .exchange import read_synthetic_exchange
from .temporal import build_temporal_fixture, run_temporal, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated synthetic PyTorch laboratory.")
    parser.add_argument("command", choices=["train", "temporal"])
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.add_argument("--stdin", action="store_true", help="Read the internal synthetic exchange for temporal evaluation.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    args = parser.parse_args()
    if args.stdin and args.command != "temporal":
        parser.error("--stdin is supported only by temporal --synthetic.")
    if args.command == "temporal":
        dataset = read_synthetic_exchange(sys.stdin) if args.stdin else build_temporal_fixture()
        report = run_temporal(*dataset)
        path = save_report(report, args.output)
        print(json.dumps({"report": str(path), **report}, indent=2, allow_nan=False))
        return
    snapshot, model = train()
    path = publish(snapshot, model, args.output)
    print(json.dumps({"mode": "synthetic", "snapshot": str(path), "training": snapshot["training"], "evaluation": snapshot["evaluation"], "promotionAllowed": False}, indent=2))


if __name__ == "__main__":
    main()
