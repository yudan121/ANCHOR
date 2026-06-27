from __future__ import annotations

import torch
from torch import nn

class QueryTeacherFeatureStudent(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        protein_dim: int,
        n_batches: int,
        n_labels: int,
        u_dim: int = 32,
        hidden_dim: int = 256,
        batch_embed_dim: int = 8,
        input_dropout: float = 0.15,
        protein_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.input_dropout = float(input_dropout)
        self.protein_dropout = float(protein_dropout)
        self.batch_embedding = nn.Embedding(max(int(n_batches), 1), int(batch_embed_dim))
        input_dim = int(z_dim) + int(protein_dim) + int(batch_embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, u_dim),
        )
        self.z_decoder = nn.Sequential(
            nn.Linear(u_dim + batch_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.protein_decoder = nn.Sequential(
            nn.Linear(u_dim + batch_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, protein_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(u_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, n_labels),
        )

    def _drop_features(self, z: torch.Tensor, protein: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return z, protein
        if self.input_dropout > 0:
            z_mask = torch.rand_like(z).ge(self.input_dropout).float()
            z = z * z_mask / max(1.0 - self.input_dropout, 1e-6)
        if self.protein_dropout > 0 and protein.numel() > 0:
            p_mask = torch.rand_like(protein).ge(self.protein_dropout).float()
            protein = protein * p_mask / max(1.0 - self.protein_dropout, 1e-6)
        return z, protein

    def forward(self, z: torch.Tensor, protein: torch.Tensor, batch_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        z_in, protein_in = self._drop_features(z, protein)
        batch_embed = self.batch_embedding(batch_idx)
        x = torch.cat([z_in, protein_in, batch_embed], dim=-1)
        u = self.encoder(x)
        dec_in = torch.cat([u, batch_embed], dim=-1)
        return {
            "u": u,
            "z_recon": self.z_decoder(dec_in),
            "protein_recon": self.protein_decoder(dec_in),
            "logits": self.classifier(u),
        }
