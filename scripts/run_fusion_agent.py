import os, sys
import argparse
import json

import torch

from agent.fusion_search_agent import FusionSearchAgent
import project_dirs as pdir

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_features(path: str) -> dict:
    data = torch.load(path, map_location="cpu")

    required = {
        "uni2",
        "virchow2",
        "optimus",
        "meta",
        "labels",
    }

    missing = required.difference(data.keys())

    if missing:
        raise KeyError(
            f"Feature file {path} is missing keys: {sorted(missing)}"
        )

    return data


parser = argparse.ArgumentParser(description='Run Embedding Fusion Agent')
parser.add_argument("--train", type=str, default="train.pt")
parser.add_argument("--val", type=str,   default="val.pt")
parser.add_argument("--test", type=str,   default="test.pt")
parser.add_argument("--output_dir", type=str, default=pdir.AGENT_DIR)
parser.add_argument("--max-trials", type=int, default=16)
parser.add_argument("--fold", type=int, default=0)

def main() -> None:

    args = parser.parse_args()

    print(f"Agent Running for Fold: {args.fold}")

    train_data = load_features(os.path.join(pdir.ARTIFACTS_DIR, "Fusion_Data", f"Fold_{args.fold}", args.train))
    val_data = load_features(os.path.join(pdir.ARTIFACTS_DIR, "Fusion_Data", f"Fold_{args.fold}", args.val))

    dims = {
        "uni2": train_data["uni2"].shape[1],
        "virchow2": train_data["virchow2"].shape[1],
        "optimus": train_data["optimus"].shape[1],
        "meta": train_data["meta"].shape[1],
    }

    assert dims == {
        "uni2": 1024,
        "virchow2": 1024,
        "optimus": 1024,
        "meta": 2,
    }, f"Unexpected input dimensions: {dims}"

    agent_output_dir = os.path.join(args.output_dir, "fusion_agent", f"Fold_{args.fold}")
    os.makedirs(agent_output_dir, exist_ok=True)

    agent = FusionSearchAgent(
        dims=dims,
        output_dir=agent_output_dir,
        device="cuda",
        max_trials=args.max_trials,
        patience=15,
        min_improvement=0.001,
    )

    summary = agent.run(train_data, val_data)

    print("\nBest fusion strategy:")
    print(json.dumps(summary["best_trial"], indent=2))


if __name__ == "__main__":
    main()