import copy
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from fusion.fusion_models import (
    STREAMS,
    EarlyConcatFusion,
    GatedFusion,
    SingleStreamClassifier,
)
from fusion.trainers import (
    predict_fusion_model,
    set_seed,
    train_fusion_model,
)


StrategyName = Literal[
    "early_concat",
    "late_prob_average",
    "late_logit_average",
    "late_weighted_vote",
    "stacking",
    "gated_fusion",
]

@dataclass(frozen=True)
class TrialConfig:
    strategy: StrategyName
    l2_norm: bool = True
    projection_dim: int = 384
    hidden_dim: int = 512
    dropout: float = 0.3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 16
    seed: int = 42

criterion = None

class FusionSearchAgent:
    """
    Deterministic, sequential fusion-search agent.

    Search phases:
    1. Coverage: evaluate every required strategy family.
    2. Adaptation: propose a small number of configurations based on the
       currently strongest validation result.
    3. Stopping: end at max_trials or after patience non-material trials.

    Notes:
    - Model selection must use train/validation only.
    - Do not load or evaluate test data in this class.
    - End-to-end PyTorch strategies are checkpointed:
      early_concat and gated_fusion.
    - Late fusion and stacking are evaluated but are not yet saved as one
      PyTorch checkpoint because they involve multiple base classifiers and,
      for stacking, a scikit-learn meta-classifier.
    """

    def __init__(
        self,
        dims: dict[str, int],
        output_dir: str | Path,
        device: str = "cuda",
        max_trials: int = 25,
        patience: int = 10,
        min_improvement: float = 0.02,
    ) -> None:
        self.dims = dims
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        self.max_trials = max_trials
        self.patience = patience
        self.min_improvement = min_improvement

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.decision_log_path = self.output_dir / "decision_log.jsonl"
        self.results_path = self.output_dir / "trial_results.json"

        self.trials: list[dict] = []
        self.tried: set[str] = set()

        self.best_trial: dict | None = None
        self.best_neural_trial: dict | None = None

        self.no_material_gain_count = 0

    def _key(self, config: TrialConfig) -> str:
        return json.dumps(asdict(config), sort_keys=True)

    def _write_log(self, record: dict) -> None:
        with self.decision_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def _checkpoint_path(
        self,
        trial_id: int,
        strategy: str,
    ) -> Path:
        return self.checkpoint_dir / (
            f"trial_{trial_id:03d}_{strategy}.pt"
        )

    def _save_torch_checkpoint(
        self,
        model: torch.nn.Module,
        config: TrialConfig,
        trial_id: int,
        val_balanced_accuracy: float,
        extra: dict | None = None,
    ) -> Path:
        """
        Save model configuration and CPU-cloned state dictionary.

        CPU clones prevent the checkpoint object from holding GPU references.
        """
        checkpoint_path = self._checkpoint_path(
            trial_id=trial_id,
            strategy=config.strategy,
        )

        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        checkpoint = {
            "created_at": datetime.now().isoformat(),
            "trial_id": trial_id,
            "strategy": config.strategy,
            "config": asdict(config),
            "input_dims": self.dims,
            "num_classes": 7,
            "val_balanced_accuracy": float(val_balanced_accuracy),
            "model_state_dict": state_dict,
            "extra": extra or {},
        }

        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def _promote_best_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Path:
        best_path = self.checkpoint_dir / "best_validation_model.pt"

        shutil.copy2(checkpoint_path, best_path)

        return best_path

    def _initial_configs(self) -> list[TrialConfig]:
        """
        Required coverage trials.

        Search includes:
        - Early concatenation, with and without L2 normalization.
        - Dimensionality reduction through learned per-stream projections.
        - Late probability and logit averaging.
        - Validation-weighted late voting.
        - Leakage-safe stacking.
        - Learned gated fusion.
        """
        common = {
            "epochs": 100,
            "batch_size": 16,
            "seed": 42,
        }

        return [
            TrialConfig(
                strategy="early_concat",
                l2_norm=False,
                projection_dim=None,
                **common,
            ),
            TrialConfig(
                strategy="early_concat",
                l2_norm=True,
                projection_dim=None,
                **common,
            ),
            TrialConfig(
                strategy="early_concat",
                l2_norm=True,
                projection_dim=384,
                **common,
            ),
            TrialConfig(
                strategy="late_prob_average",
                **common,
            ),
            TrialConfig(
                strategy="late_logit_average",
                **common,
            ),
            TrialConfig(
                strategy="late_weighted_vote",
                **common,
            ),
            TrialConfig(
                strategy="stacking",
                **common,
            ),
            TrialConfig(
                strategy="gated_fusion",
                l2_norm=True,
                projection_dim=768,
                **common,
            ),
        ]

    def propose_next(self) -> tuple[TrialConfig | None, str]:
        """
        Propose the next configuration from observed outcomes.

        First complete coverage of mandatory strategy families. Then test only
        a few variants of the strongest strategy family to avoid excessive
        adaptation to a small validation set.
        """
        for config in self._initial_configs():
            if self._key(config) not in self.tried:
                return (
                    config,
                    "Coverage phase: evaluate a required fusion strategy "
                    "before spending additional trials on one family.",
                )

        if self.best_trial is None:
            return None, "No valid trial completed."

        winner = TrialConfig(**self.best_trial["config"])

        candidates: list[tuple[TrialConfig, str]] = []

        if winner.strategy == "early_concat":
            candidates.extend([
                (
                    TrialConfig(
                        strategy="early_concat",
                        l2_norm=True,
                        projection_dim=128,
                        hidden_dim=256,
                        dropout=0.50,
                        learning_rate=1e-4,
                        weight_decay=1e-4,
                        epochs=20,
                        batch_size=16,
                        seed=43,
                    ),
                    "Early concatenation is strongest so far; test lower "
                    "dimensional projections and stronger regularization.",
                ),
                (
                    TrialConfig(
                        strategy="early_concat",
                        l2_norm=True,
                        projection_dim=256,
                        hidden_dim=384,
                        dropout=0.40,
                        learning_rate=1e-4,
                        weight_decay=1e-4,
                        epochs=20,
                        batch_size=16,
                        seed=43,
                    ),
                    "Early concatenation is strongest so far; test whether "
                    "moderate compression preserves complementary features.",
                ),
            ])

        elif winner.strategy in {
            "late_prob_average",
            "late_logit_average",
            "late_weighted_vote",
            "stacking",
        }:
            candidates.extend([
                (
                    TrialConfig(
                        strategy="stacking",
                        dropout=0.50,
                        learning_rate=1e-4,
                        weight_decay=1e-4,
                        epochs=20,
                        batch_size=16,
                        seed=43,
                    ),
                    "Late fusion is strongest so far; test regularized "
                    "stacking with train-only out-of-fold features.",
                ),
                (
                    TrialConfig(
                        strategy="gated_fusion",
                        l2_norm=True,
                        projection_dim=128,
                        hidden_dim=256,
                        dropout=0.50,
                        learning_rate=1e-4,
                        weight_decay=1e-4,
                        epochs=20,
                        batch_size=16,
                        seed=43,
                    ),
                    "Late fusion is strong; test whether patient-specific "
                    "learned stream weighting improves it.",
                ),
            ])

        elif winner.strategy == "gated_fusion":
            candidates.extend([
                (
                    TrialConfig(
                        strategy="gated_fusion",
                        l2_norm=True,
                        projection_dim=256,
                        hidden_dim=384,
                        dropout=0.50,
                        learning_rate=1e-4,
                        weight_decay=1e-4,
                        epochs=20,
                        batch_size=16,
                        seed=43,
                    ),
                    "Gated fusion is strongest so far; reduce capacity and "
                    "increase regularization for small-sample stability.",
                ),
                (
                    TrialConfig(
                        strategy="gated_fusion",
                        l2_norm=True,
                        projection_dim=384,
                        hidden_dim=768,
                        dropout=0.50,
                        learning_rate=1e-4,
                        weight_decay=1e-4,
                        epochs=20,
                        batch_size=16,
                        seed=44,
                    ),
                    "Gated fusion is strongest so far; repeat with a second "
                    "seed to assess seed sensitivity.",
                ),
            ])

        for config, reason in candidates:
            if self._key(config) not in self.tried:
                return config, reason

        return (
            None,
            "All targeted follow-up trials are complete; stop rather than "
            "continue adapting decisions to the validation set.",
        )

    def _fit_single_stream(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_eval: torch.Tensor,
        config: TrialConfig,
    ) -> np.ndarray:
        """
        Train one small classifier for a single stream.

        Used for late fusion and stacking. L2 normalization is applied to each
        stream because the modalities have very different raw dimensions and
        feature scales.
        """
        set_seed(config.seed)

        x_train = F.normalize(x_train.float(), p=2, dim=1)
        x_eval = F.normalize(x_eval.float(), p=2, dim=1)

        model = SingleStreamClassifier(
            input_dim=x_train.shape[1],
            hidden_dim=128,
            dropout=config.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        class_counts = torch.bincount(y_train, minlength=7).float()
        class_weights = class_counts.sum() / (
            7 * class_counts.clamp_min(1)
        )

        global criterion
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )

        x_train = x_train.to(self.device)
        y_train = y_train.long().to(self.device)
        x_eval = x_eval.to(self.device)

        model.train()

        for _ in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)

            logits = model(x_train)
            loss = criterion(logits, y_train)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():
            logits = model(x_eval).detach().cpu().numpy()

        return logits

    def _late_fusion_predictions(
        self,
        train_data: dict,
        eval_data: dict,
        config: TrialConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train one classifier per stream and predict evaluation samples.

        Returns:
            logits: shape [N_eval, 4, 7]
            probabilities: shape [N_eval, 4, 7]
        """
        all_logits = []

        for stream in STREAMS:
            logits = self._fit_single_stream(
                x_train=train_data[stream],
                y_train=train_data["labels"],
                x_eval=eval_data[stream],
                config=config,
            )

            all_logits.append(logits)

        logits_array = np.stack(all_logits, axis=1)

        probabilities = torch.softmax(
            torch.tensor(logits_array),
            dim=-1,
        ).numpy()

        return logits_array, probabilities

    def _oof_features(
        self,
        train_data: dict,
        config: TrialConfig,
        n_splits: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create leakage-safe out-of-fold base-model probability features.

        Each training patient receives predictions from base classifiers that
        were trained without that patient's fold. These out-of-fold features
        are then used to train the logistic-regression stacker.
        """
        labels = train_data["labels"].detach().cpu().numpy()
        n_samples = len(labels)

        min_class_count = np.bincount(labels).min()

        if min_class_count < n_splits:
            raise ValueError(
                f"Cannot use {n_splits}-fold stratified stacking: "
                f"the smallest class has only {min_class_count} samples."
            )

        oof_features = np.zeros(
            (n_samples, len(STREAMS) * 7),
            dtype=np.float32,
        )

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.seed,
        )

        for train_indices, holdout_indices in splitter.split(
            np.zeros(n_samples),
            labels,
        ):
            fold_train = {
                key: value[train_indices]
                for key, value in train_data.items()
                if isinstance(value, torch.Tensor)
            }

            fold_holdout = {
                key: value[holdout_indices]
                for key, value in train_data.items()
                if isinstance(value, torch.Tensor)
            }

            _, fold_probabilities = self._late_fusion_predictions(
                train_data=fold_train,
                eval_data=fold_holdout,
                config=config,
            )

            oof_features[holdout_indices] = fold_probabilities.reshape(
                len(holdout_indices),
                -1,
            )

        return oof_features, labels

    def _evaluate(
        self,
        config: TrialConfig,
        train_data: dict,
        val_data: dict,
    ) -> dict:
        """
        Evaluate one proposed configuration on validation data only.

        Returns a `trained_model` only for end-to-end neural architectures.
        For late fusion and stacking it returns None because those approaches
        involve multiple separately trained base models.
        """
        model: torch.nn.Module | None = None
        info: dict = {}
        fusion_details: dict = {}

        if config.strategy == "early_concat":
            model = EarlyConcatFusion(
                dims=self.dims,
                l2_norm=config.l2_norm,
                projection_dim=config.projection_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )

            model, info = train_fusion_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                device=self.device,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                epochs=config.epochs,
                batch_size=config.batch_size,
                seed=config.seed,
            )

            y_true, y_prob, mean_loss = predict_fusion_model(
                model=model,
                data=val_data,
                batch_size=config.batch_size,
                device=self.device,
                criterion=criterion,
            )

        elif config.strategy == "gated_fusion":
            model = GatedFusion(
                dims=self.dims,
                common_dim=config.projection_dim or 256,
                gate_hidden_dim=config.hidden_dim,
                classifier_hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                l2_norm=config.l2_norm,
            )

            model, info = train_fusion_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                device=self.device,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                epochs=config.epochs,
                batch_size=config.batch_size,
                seed=config.seed,
            )

            y_true, y_prob, mean_loss = predict_fusion_model(
                model=model,
                data=val_data,
                batch_size=config.batch_size,
                device=self.device,
            )

        else:
            y_true = val_data["labels"].detach().cpu().numpy()

            val_logits, val_probs = self._late_fusion_predictions(
                train_data=train_data,
                eval_data=val_data,
                config=config,
            )

            if config.strategy == "late_prob_average":
                y_prob = val_probs.mean(axis=1)

            elif config.strategy == "late_logit_average":
                average_logits = val_logits.mean(axis=1)

                y_prob = torch.softmax(
                    torch.tensor(average_logits),
                    dim=1,
                ).numpy()

            elif config.strategy == "late_weighted_vote":
                stream_scores = []

                for stream_index in range(len(STREAMS)):
                    stream_prediction = val_probs[
                        :,
                        stream_index,
                    ].argmax(axis=1)

                    stream_scores.append(
                        balanced_accuracy_score(
                            y_true,
                            stream_prediction,
                        )
                    )

                fusion_weights = np.asarray(
                    stream_scores,
                    dtype=np.float64,
                )

                # A stream below random balanced accuracy should not dominate.
                fusion_weights = np.clip(
                    fusion_weights - (1.0 / 7.0),
                    1e-4,
                    None,
                )

                fusion_weights /= fusion_weights.sum()

                y_prob = np.average(
                    val_probs,
                    axis=1,
                    weights=fusion_weights,
                )

                fusion_details["stream_weights"] = {
                    stream: float(weight)
                    for stream, weight in zip(STREAMS, fusion_weights)
                }

            elif config.strategy == "stacking":
                oof_features, oof_labels = self._oof_features(
                    train_data=train_data,
                    config=config,
                    n_splits=3,
                )

                val_features = val_probs.reshape(len(y_true), -1)

                stacker = LogisticRegression(
                    C=0.1,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=config.seed,
                )

                stacker.fit(oof_features, oof_labels)
                y_prob = stacker.predict_proba(val_features)

                fusion_details["stacker_C"] = 0.1
                fusion_details["stacker_feature_dim"] = int(
                    val_features.shape[1]
                )

            else:
                raise ValueError(
                    f"Unsupported strategy: {config.strategy}"
                )

        y_pred = y_prob.argmax(axis=1)

        val_balanced_accuracy = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        return {
            "val_balanced_accuracy": float(val_balanced_accuracy),
            "val_predictions": y_pred.tolist(),
            "val_labels": y_true.tolist(),
            "details": {
                **info,
                **fusion_details,
            },
            "trained_model": model,
        }

    def run(
        self,
        train_data: dict,
        val_data: dict,
    ) -> dict:
        """
        Execute the sequential validation-only fusion search.

        Every successful neural trial is checkpointed. The best neural trial is
        copied to `best_validation_model.pt`. If the overall best trial is a
        late-fusion strategy, `best_trial` records it, but it requires a future
        bundle-saving implementation for final test inference.
        """
        while len(self.trials) < self.max_trials:
            config, proposal_reason = self.propose_next()

            if config is None:
                break

            trial_id = len(self.trials) + 1

            self.tried.add(self._key(config))

            result = self._evaluate(
                config=config,
                train_data=train_data,
                val_data=val_data,
            )

            score = result["val_balanced_accuracy"]

            previous_best = (
                -1.0
                if self.best_trial is None
                else self.best_trial["val_balanced_accuracy"]
            )

            is_new_best = score > previous_best

            is_material_gain = score >= (
                previous_best + self.min_improvement
            )

            if is_new_best:
                self.best_trial = {
                    "trial_id": trial_id,
                    "strategy": config.strategy,
                    "config": asdict(config),
                    "val_balanced_accuracy": float(score),
                    "checkpoint_path": None,
                }

            if is_material_gain:
                self.no_material_gain_count = 0
            else:
                self.no_material_gain_count += 1

            trial_record = {
                "trial_id": trial_id,
                "proposal_reason": proposal_reason,
                "config": asdict(config),
                "val_balanced_accuracy": float(score),
                "previous_best_balanced_accuracy": float(previous_best),
                "is_new_best": bool(is_new_best),
                "is_material_gain": bool(is_material_gain),
                "details": result["details"],
                "checkpoint_path": None,
            }

            trained_model = result.get("trained_model")

            if trained_model is not None:
                trial_checkpoint_path = self._save_torch_checkpoint(
                    model=trained_model,
                    config=config,
                    trial_id=trial_id,
                    val_balanced_accuracy=score,
                    extra={
                        "proposal_reason": proposal_reason,
                        "is_new_best_overall": is_new_best,
                        "is_material_gain": is_material_gain,
                    },
                )

                trial_record["checkpoint_path"] = str(
                    trial_checkpoint_path
                )

                neural_previous_best = (
                    -1.0
                    if self.best_neural_trial is None
                    else self.best_neural_trial[
                        "val_balanced_accuracy"
                    ]
                )

                is_new_neural_best = score > neural_previous_best

                if is_new_neural_best:
                    best_checkpoint_path = self._promote_best_checkpoint(
                        trial_checkpoint_path
                    )

                    self.best_neural_trial = {
                        "trial_id": trial_id,
                        "strategy": config.strategy,
                        "config": asdict(config),
                        "val_balanced_accuracy": float(score),
                        "checkpoint_path": str(best_checkpoint_path),
                    }

                    # If this neural model is also globally best, associate the
                    # global best record with its promoted checkpoint.
                    if is_new_best and self.best_trial is not None:
                        self.best_trial["checkpoint_path"] = str(
                            best_checkpoint_path
                        )

            print(
                f"[Trial {trial_id:02d}/{self.max_trials}] "
                f"strategy={config.strategy} | "
                f"val_bal_acc={score:.4f} | "
                f"best_before={previous_best:.4f} | "
                f"new_best={is_new_best}"
            )

            print(f"  Reason: {proposal_reason}")

            if trial_record["checkpoint_path"] is not None:
                print(
                    "  Saved model: "
                    f"{trial_record['checkpoint_path']}"
                )

            if (
                trained_model is not None
                and self.best_neural_trial is not None
                and self.best_neural_trial["trial_id"] == trial_id
            ):
                print(
                    "  Promoted model: "
                    f"{self.best_neural_trial['checkpoint_path']}"
                )

            self.trials.append(trial_record)
            self._write_log(trial_record)

            if self.no_material_gain_count >= self.patience:
                stop_record = {
                    "event": "search_stopped",
                    "reason": (
                        "No validation balanced-accuracy gain of at least "
                        f"{self.min_improvement:.3f} over "
                        f"{self.patience} consecutive trials."
                    ),
                    "completed_trials": len(self.trials),
                    "best_trial": self.best_trial,
                    "best_neural_trial": self.best_neural_trial,
                }

                self._write_log(stop_record)

                print(
                    "\nSearch stopped early: "
                    f"{self.patience} consecutive non-material trials."
                )

                break

        summary = {
            "trial_budget": self.max_trials,
            "completed_trials": len(self.trials),
            "minimum_material_improvement": self.min_improvement,
            "patience": self.patience,
            "best_trial": self.best_trial,
            "best_neural_trial": self.best_neural_trial,
            "trials": self.trials,
        }

        with self.results_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        return summary