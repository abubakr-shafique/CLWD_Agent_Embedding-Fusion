import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PatientFeatureDataset(Dataset):
    def __init__(self, data: dict) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data["labels"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "uni2": self.data["uni2"][index],
            "virchow2": self.data["virchow2"][index],
            "optimus": self.data["optimus"][index],
            "meta": self.data["meta"][index],
            "labels": self.data["labels"][index],
        }


def make_loader(
    data: dict,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        PatientFeatureDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def class_weights(labels: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp_min(1))


@torch.no_grad()
def predict_fusion_model(
    model: nn.Module,
    data: dict,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    loader = make_loader(data, batch_size=batch_size, shuffle=False)

    model.eval()
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {name: x.float().to(device) for name, x in batch.items()}

        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)

        if criterion is not None:
            loss = criterion(logits, labels.long())
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        all_labels.append(labels.cpu())
        all_probabilities.append(probabilities.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probabilities).numpy()
    mean_loss = total_loss / total_samples if criterion is not None and total_samples > 0 else None

    return y_true, y_prob, mean_loss


def train_fusion_model(
    model: nn.Module,
    train_data: dict,
    val_data: dict,
    device: torch.device,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[nn.Module, dict]:
    set_seed(seed)

    train_loader = make_loader(train_data, batch_size, shuffle=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    current_lr = 0
    weights = class_weights(train_data["labels"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    early_stop = True
    tolerance = 50
    es = 0

    best_state = None
    best_val_score = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_samples = 0
        all_train_labels = []
        all_train_preds = []

        for batch in train_loader:
            labels = batch.pop("labels").long().to(device)
            batch = {name: x.float().to(device) for name, x in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_samples += labels.size(0)
            all_train_labels.append(labels.detach().cpu())
            all_train_preds.append(logits.detach().argmax(dim=1).cpu())

        train_loss = train_loss_sum / train_samples
        y_train = torch.cat(all_train_labels).numpy()
        train_pred = torch.cat(all_train_preds).numpy()
        train_bal_acc = balanced_accuracy_score(y_train, train_pred)

        y_val, prob_val, val_loss = predict_fusion_model(
            model,
            val_data,
            batch_size,
            device,
            criterion=criterion,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr > 1e-6:
            scheduler.step()

        val_pred = prob_val.argmax(axis=1)
        val_bal_acc = balanced_accuracy_score(y_val, val_pred)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_balanced_accuracy": float(train_bal_acc),
            "val_balanced_accuracy": float(val_bal_acc),
        })

        if val_bal_acc > best_val_score:
            best_val_score = float(val_bal_acc)
            best_state = copy.deepcopy(model.state_dict())
            es = 0

        es = es + 1
        if es >= tolerance and early_stop:
            break

    model.load_state_dict(best_state)

    return model, {
        "best_val_balanced_accuracy": best_val_score,
        "history": history,
    }