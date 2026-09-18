import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 7
STREAMS = ("uni2", "virchow2", "optimus", "meta")


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1, eps=1e-8)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyConcatFusion(nn.Module):
    """
    Concatenate UNI2, Virchow2, H-OPTIMUS-1, age/sex.
    Optional L2 normalization and optional learned dimension reduction.
    """
    def __init__(
        self,
        dims: dict[str, int],
        l2_norm: bool = True,
        projection_dim: int | None = None,
        hidden_dim: int = 1024,
        dropout: float = 0.40,
    ) -> None:
        super().__init__()

        self.l2_norm = l2_norm
        self.projectors = nn.ModuleDict()
        fused_dim = 0

        for stream in STREAMS:
            dim = dims[stream]

            if projection_dim is None:
                self.projectors[stream] = nn.Identity()
                fused_dim += dim
            else:
                self.projectors[stream] = nn.Sequential(
                    nn.Linear(dim, projection_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                fused_dim += projection_dim

        self.head = ClassifierHead(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_features: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = []

        for stream in STREAMS:
            x = batch[stream]

            if self.l2_norm:
                x = l2_normalize(x)

            features.append(self.projectors[stream](x))

        fused_embedding = torch.cat(features, dim=1)
        logits = self.head(fused_embedding)

        if return_features:
            return logits, fused_embedding

        return logits


class GatedFusion(nn.Module):
    """
    Intermediate learned fusion.

    Every modality is projected to a shared latent space. A gating network
    produces one weight per modality for every patient.
    """
    def __init__(
        self,
        dims: dict[str, int],
        common_dim: int = 256,
        gate_hidden_dim: int = 256,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.40,
        l2_norm: bool = True,
    ) -> None:
        super().__init__()

        self.l2_norm = l2_norm

        self.projectors = nn.ModuleDict({
            stream: nn.Sequential(
                nn.Linear(dims[stream], common_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for stream in STREAMS
        })

        self.gate = nn.Sequential(
            nn.Linear(common_dim * len(STREAMS), gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, len(STREAMS)),
        )

        self.head = ClassifierHead(
            input_dim=common_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_features: bool = False,
        ) -> torch.Tensor | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        ]:
        projected = []

        for stream in STREAMS:
            x = batch[stream]

            if self.l2_norm:
                x = l2_normalize(x)

            projected.append(self.projectors[stream](x))

        gate_input = torch.cat(projected, dim=1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=1)

        modality_tensor = torch.stack(projected, dim=1)

        fused_embedding = (
            modality_tensor * gate_weights.unsqueeze(-1)
        ).sum(dim=1)

        logits = self.head(fused_embedding)

        if return_features:
            return logits, fused_embedding, gate_weights

        return logits


class SingleStreamClassifier(nn.Module):
    """
    Used for late fusion. One model is trained for each modality:
    UNI2, Virchow2, H-OPTIMUS-1, and metadata.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.head = ClassifierHead(input_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)