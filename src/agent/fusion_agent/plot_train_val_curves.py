"""Plot train/validation accuracy and loss curves for the best neural trial.

Reads the best trial id from trial_results.json ("best_neural_trial"), finds the
matching record in decision_log.jsonl, and plots per-epoch curves in one figure
with 1 row / 2 columns:
    subplot 1 -> train and validation accuracy
    subplot 2 -> train and validation loss

Note: the current decision_log.jsonl history entries only contain
"epoch" and "val_balanced_accuracy". If train accuracy / losses are not logged,
those curves are skipped and the subplot is annotated accordingly. The metric
key lookup below will automatically pick up train/loss data once your training
loop logs it (e.g. train_balanced_accuracy, train_loss, val_loss).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

TRIAL_RESULTS_PATH = "Fold_2\\trial_results.json"
DECISION_LOG_PATH = "Fold_2\\decision_log.jsonl"
OUT_PNG = "Fold_2\\best_trial_curves.png"

# Candidate key names for each metric, in priority order.
ACC_TRAIN_KEYS = ["train_balanced_accuracy", "train_accuracy", "train_acc"]
ACC_VAL_KEYS = ["val_balanced_accuracy", "val_accuracy", "val_acc"]
LOSS_TRAIN_KEYS = ["train_loss", "loss"]
LOSS_VAL_KEYS = ["val_loss", "valid_loss", "validation_loss"]


def get_best_trial_id(trial_results_path):
    with open(trial_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    best = results.get("best_neural_trial")
    if not best:
        raise KeyError(f"'best_neural_trial' not found in {trial_results_path}")
    return best["trial_id"], best.get("config", {})


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_number}: {e}")
    return records


def configs_match(cfg_a, cfg_b):
    """Best trial_id repeats across appended sweeps, so disambiguate by config."""
    keys = ["strategy", "l2_norm", "projection_dim", "hidden_dim",
            "dropout", "learning_rate", "weight_decay", "batch_size", "seed"]
    return all(cfg_a.get(k) == cfg_b.get(k) for k in keys)


def find_trial_record(records, trial_id, best_config):
    matches = [r for r in records if r.get("trial_id") == trial_id]
    if not matches:
        raise ValueError(f"No record with trial_id={trial_id} in the decision log")
    if best_config:
        for r in reversed(matches):  # log appends; latest run wins on ties
            if configs_match(r.get("config", {}), best_config):
                return r
    return matches[-1]


def extract_metric(history, key_candidates):
    key = next((k for k in key_candidates
                if any(k in h for h in history)), None)
    if key is None:
        return [], []
    epochs = [h["epoch"] for h in history if key in h]
    values = [h[key] for h in history if key in h]
    return epochs, values


def plot_curves(history, trial_id, strategy, out_png):
    epochs = [h["epoch"] for h in history]

    tr_ep, tr_acc = extract_metric(history, ACC_TRAIN_KEYS)
    va_ep, va_acc = extract_metric(history, ACC_VAL_KEYS)
    trl_ep, trl = extract_metric(history, LOSS_TRAIN_KEYS)
    val_ep, val = extract_metric(history, LOSS_VAL_KEYS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    fig.suptitle(f"Best Neural Trial {trial_id} ({strategy})", fontsize=13)

    ax1, ax2 = axes
    if tr_acc:
        ax1.plot(tr_ep, tr_acc, "-o", ms=4, label="Train Accuracy")
    if va_acc:
        ax1.plot(va_ep, va_acc, "-s", ms=4, color="darkorange",
                 label="Val Accuracy (balanced)")
    if not (tr_acc or va_acc):
        ax1.text(0.5, 0.5, "No accuracy data in history",
                 ha="center", va="center", transform=ax1.transAxes)
    elif not tr_acc:
        ax1.text(0.5, 0.02, "Train accuracy not logged in decision_log.jsonl",
                 ha="center", va="bottom", transform=ax1.transAxes,
                 fontsize=9, color="gray")
    ax1.set_title("Accuracy vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(epochs if len(epochs) <= 25 else epochs[:: max(1, len(epochs) // 12)])
    ax1.grid(alpha=0.3)
    if tr_acc or va_acc:
        ax1.legend()

    if trl:
        ax2.plot(trl_ep, trl, "-o", ms=4, label="Train Loss")
    if val:
        ax2.plot(val_ep, val, "-s", ms=4, color="darkorange", label="Val Loss")
    if not (trl or val):
        ax2.text(0.5, 0.5,
                 "No loss data in history\n(loss is not logged in decision_log.jsonl)",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")
    ax2.set_title("Loss vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    if trl or val:
        ax2.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=200)
    print(f"Saved figure -> {Path(out_png).resolve()}")
    plt.show()


def main():
    trial_id, best_config = get_best_trial_id(TRIAL_RESULTS_PATH)
    records = load_jsonl(DECISION_LOG_PATH)
    record = find_trial_record(records, trial_id, best_config)

    history = record.get("details", {}).get("history", [])
    if not history:
        raise ValueError(f"Trial {trial_id} has no per-epoch history in details")

    strategy = record.get("config", {}).get("strategy", "unknown")
    best_val = record.get("details", {}).get("best_val_balanced_accuracy")
    print(f"Trial {trial_id} | strategy={strategy} | "
          f"epochs={len(history)} | best val balanced acc={best_val}")

    plot_curves(history, trial_id, strategy, OUT_PNG)


if __name__ == "__main__":
    main()
