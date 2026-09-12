import argparse
import os, sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
import project_dirs as pdir
from fusion.fusion_models import EarlyConcatFusion, GatedFusion
import utils.eval_utils as eval_utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_0", "CLWD_OneSlide-train.csv"))
train_Labels = train_csv_Data["Benchmark_Label_7class"].tolist()

label_names = sorted(set(train_Labels))
n_classes = len(label_names)
label_to_index = {label: idx for idx, label in enumerate(label_names)}
index_to_label = {idx: label for idx, label in enumerate(label_names)}

CLASS_NAMES = label_names


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

class TestFeatureDataset(Dataset):
    def __init__(self, data: dict) -> None:
        required = {"uni2", "virchow2", "optimus", "meta", "labels"}
        missing = required.difference(data.keys())

        if missing:
            raise KeyError(
                f"Test data is missing required keys: {sorted(missing)}"
            )

        n = len(data["labels"])

        for key in required:
            if len(data[key]) != n:
                raise ValueError(
                    f"Length mismatch: '{key}' has {len(data[key])}, "
                    f"but labels has {n}."
                )

        self.data = data
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "uni2": self.data["uni2"][index],
            "virchow2": self.data["virchow2"][index],
            "optimus": self.data["optimus"][index],
            "meta": self.data["meta"][index],
            "labels": self.data["labels"][index],
        }


def make_loader(test_data: dict, batch_size: int) -> DataLoader:
    return DataLoader(
        TestFeatureDataset(test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def build_model_from_checkpoint(
    checkpoint: dict,
) -> torch.nn.Module:
    strategy = checkpoint["strategy"]
    config = checkpoint["config"]
    dims = checkpoint["input_dims"]

    if strategy == "early_concat":
        model = EarlyConcatFusion(
            dims=dims,
            l2_norm=config["l2_norm"],
            projection_dim=config["projection_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )

    elif strategy == "gated_fusion":
        model = GatedFusion(
            dims=dims,
            common_dim=config["projection_dim"] or 256,
            gate_hidden_dim=config["hidden_dim"],
            classifier_hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            l2_norm=config["l2_norm"],
        )

    else:
        raise ValueError(
            f"Checkpoint strategy '{strategy}' is not a single "
            "end-to-end PyTorch fusion model. "
            "Use a dedicated evaluation bundle for late fusion/stacking."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


@torch.no_grad()
def predict_and_save_artifacts(
    model: torch.nn.Module,
    test_data: dict,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    Returns:
        results = {
            "True_label": Tensor[N],
            "Pred_label": Tensor[N],
            "logits": Tensor[N, 7],
            "fused_embed": Tensor[N, fused_dimension],
            "probabilities": Tensor[N, 7],
            "uni2_embed": Tensor[N, 1024],
            "virchow2_embed": Tensor[N, 1024],
            "optimus_embed": Tensor[N, 1024],
            "metadata": Tensor[N, 2],
            "patient_ids": list[str],
            "gate_weights": Tensor[N, 4]  # gated fusion only
        }
    """
    loader = make_loader(test_data, batch_size)

    model.eval()

    all_true_labels = []
    all_pred_labels = []
    all_logits = []
    all_probabilities = []
    all_fused_embeddings = []

    all_uni2 = []
    all_virchow2 = []
    all_optimus = []
    all_metadata = []
    all_gate_weights = []

    is_gated_model = isinstance(model, GatedFusion)

    for batch in loader:
        true_labels = batch.pop("labels").long()

        inputs = {
            key: value.float().to(device)
            for key, value in batch.items()
        }

        if is_gated_model:
            logits, fused_embed, gate_weights = model(
                inputs,
                return_features=True,
            )

            all_gate_weights.append(
                gate_weights.detach().cpu()
            )

        else:
            logits, fused_embed = model(
                inputs,
                return_features=True,
            )

        probabilities = torch.softmax(logits, dim=1)
        pred_labels = torch.argmax(probabilities, dim=1)

        all_true_labels.append(true_labels.cpu())
        all_pred_labels.append(pred_labels.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_probabilities.append(probabilities.detach().cpu())
        all_fused_embeddings.append(fused_embed.detach().cpu())

        all_uni2.append(inputs["uni2"].detach().cpu())
        all_virchow2.append(inputs["virchow2"].detach().cpu())
        all_optimus.append(inputs["optimus"].detach().cpu())
        all_metadata.append(inputs["meta"].detach().cpu())

    results = {
        "True_label": torch.cat(all_true_labels, dim=0),
        "Pred_label": torch.cat(all_pred_labels, dim=0),
        "logits": torch.cat(all_logits, dim=0),
        "fused_embed": torch.cat(all_fused_embeddings, dim=0),
        "probabilities": torch.cat(all_probabilities, dim=0),
        "uni2_embed": torch.cat(all_uni2, dim=0),
        "virchow2_embed": torch.cat(all_virchow2, dim=0),
        "optimus_embed": torch.cat(all_optimus, dim=0),
        "metadata": torch.cat(all_metadata, dim=0),
        "patient_ids": test_data.get(
            "ids",
            [f"test_patient_{i:03d}" for i in range(len(test_data["labels"]))],
        ),
    }

    if is_gated_model:
        results["gate_weights"] = torch.cat(
            all_gate_weights,
            dim=0,
        )

    return results


def compute_metrics(results: dict) -> dict:
    y_true = results["True_label"].numpy()
    y_pred = results["Pred_label"].numpy()
    y_prob = results["probabilities"].numpy()

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    one_hot = np.eye(len(CLASS_NAMES))[y_true]

    per_class_auroc = {}

    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_targets = one_hot[:, class_index]

        if len(np.unique(binary_targets)) < 2:
            per_class_auroc[class_name] = None
            continue

        per_class_auroc[class_name] = float(
            roc_auc_score(
                binary_targets,
                y_prob[:, class_index],
            )
        )

    valid_aurocs = [
        score
        for score in per_class_auroc.values()
        if score is not None
    ]

    macro_auroc = (
        float(np.mean(valid_aurocs))
        if valid_aurocs
        else None
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )

    return {
        "n_test_samples": int(len(y_true)),
        "balanced_accuracy": float(balanced_acc),
        "macro_auroc_ovr": macro_auroc,
        "per_class_auroc_ovr": per_class_auroc,
        "confusion_matrix": matrix.tolist(),
    }

parser = argparse.ArgumentParser(description='Evaluate Embedding Fusion Agent')
parser.add_argument("--checkpoint",type=str, default=pdir.AGENT_DIR)
parser.add_argument("--test", type=str,   default="test.pt")
parser.add_argument("--result_dir", type=str, default=pdir.RESULT_DIR)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--fold", type=int, default=0)

args = parser.parse_args()

def main() -> None:

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.result_dir, "Embedding_Fusion", f"Fold_{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_checkpoint = os.path.join(args.checkpoint, "fusion_agent", f"Fold_{args.fold}", "checkpoints", "best_validation_model.pt")
    checkpoint = torch.load(best_val_checkpoint, map_location="cpu")

    test_data = load_features(os.path.join(pdir.ARTIFACTS_DIR, "Fusion_Data", f"Fold_{args.fold}", args.test))

    model = build_model_from_checkpoint(checkpoint)
    model = model.to(device)

    results = predict_and_save_artifacts(
        model=model,
        test_data=test_data,
        batch_size=args.batch_size,
        device=device,
    )

    # print(results.keys())
    ## save in this folder
    results_summary_save_dir = os.path.join(output_dir, f"BestFusion_test_results.npy")
    eval_utils.multiclass_classification_report(y_true=results["True_label"], y_pred=results["Pred_label"], n_classes=len(CLASS_NAMES), class_names=CLASS_NAMES, save_dir=results_summary_save_dir)

    ROC_save_dir = os.path.join(output_dir, f"BestFusion_test_ROC.jpg")
    eval_utils.plot_multiclass_roc_agent(eval_history=results, classes=index_to_label, output_file=ROC_save_dir)

    accuracy_save_dir = os.path.join(output_dir, f"BestFusion_test_Overall_Accuracy.jpg")
    eval_utils.evaluate_and_plot_metrics_agent(eval_history=results, class_to_idx=label_to_index, output_file=accuracy_save_dir)

    results["checkpoint_path"] = str(args.checkpoint)
    results["strategy"] = checkpoint["strategy"]
    results["architecture_config"] = checkpoint["config"]

    output_artifact_path = output_dir / "test_results.pt"

    torch.save(
        results,
        output_artifact_path,
    )

    metrics = compute_metrics(results)

    metrics["checkpoint_path"] = str(args.checkpoint)
    metrics["strategy"] = checkpoint["strategy"]
    metrics["validation_balanced_accuracy"] = float(
        checkpoint["val_balanced_accuracy"]
    )

    with (output_dir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nTest evaluation complete")
    print(f"Strategy: {checkpoint['strategy']}")
    print(
        "Validation balanced accuracy: "
        f"{checkpoint['val_balanced_accuracy']:.4f}"
    )
    print(
        "Test balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Test macro AUROC: {metrics['macro_auroc_ovr']}")
    print(f"Saved artifacts: {output_artifact_path}")
    print(f"Saved metrics: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()