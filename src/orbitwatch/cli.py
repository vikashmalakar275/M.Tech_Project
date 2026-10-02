from __future__ import annotations

import argparse
import json
import sys

from orbitwatch.config import ExperimentConfig, project_root
from orbitwatch.data import download_dataset
from orbitwatch.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OrbitWatch spacecraft telemetry research workbench"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download and verify the original NASA benchmark mirror.")
    train = sub.add_parser(
        "train", help="Train and evaluate detectors; no test labels are used for fitting."
    )
    train.add_argument(
        "--channels", nargs="+", help="Explicit channel subset. Default: every channel."
    )
    train.add_argument("--run-id")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--max-train-windows", type=int, default=6000)
    explain = sub.add_parser("evaluate-explanations", help="Run real local-model comparison cases.")
    explain.add_argument("--run")
    explain.add_argument("--model", default="qwen2.5:3b")
    explain.add_argument("--cases", type=int, default=12)
    report = sub.add_parser(
        "report", help="Generate measured-results PDF, slides, and research artifacts."
    )
    report.add_argument("--run")
    args = parser.parse_args()
    root = project_root()
    if args.command == "download":
        print(json.dumps(download_dataset(root / "data"), indent=2))
    elif args.command == "train":
        config = ExperimentConfig(
            epochs=args.epochs, seed=args.seed, max_train_windows=args.max_train_windows
        )
        run_experiment(
            root, config, args.channels, args.run_id, progress=lambda text: print(text, flush=True)
        )
    elif args.command == "evaluate-explanations":
        from orbitwatch.explanation_eval import evaluate_explanations

        print(evaluate_explanations(root, args.run, args.model, args.cases))
    elif args.command == "report":
        from orbitwatch.submission import create_submission

        print(create_submission(root, args.run))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"OrbitWatch: {error}", file=sys.stderr)
        sys.exit(1)
