import argparse
import json
from .config import SETTINGS

def main():
    parser = argparse.ArgumentParser(description="RetailAssist reproducible local workflow")
    parser.add_argument("command", choices=["download", "prepare", "policies", "index", "train", "calibrate", "evaluate", "report", "status"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="Train an isolated tiny checkpoint; never replace the real model")
    parser.add_argument("--metadata", default="submission/group.json")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    from .training import download_models, prepare_data, train
    from .models import LocalModels
    if args.command == "download": result = download_models()
    elif args.command == "prepare": result = prepare_data()
    elif args.command == "policies":
        from .documents import policy_pdfs
        result = policy_pdfs()
    elif args.command == "index":
        from .retrieval import build_index
        result = build_index(LocalModels())
    elif args.command == "train": result = train(epochs=args.epochs, smoke=args.smoke)
    elif args.command in ["calibrate", "evaluate"]:
        from .evaluation import calibrate, evaluate
        result = calibrate() if args.command == "calibrate" else evaluate(skip_generation=args.skip_generation)
    elif args.command == "report":
        from .reporting import build_reports
        result = build_reports(args.metadata)
    else: result = LocalModels().readiness()
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__": main()
