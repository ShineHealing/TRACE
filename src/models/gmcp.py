"""GMCP: gene-module compositional pyramid."""

import torch
import torch.nn as nn


class MPN(nn.Module):
    """Predict module-prior logits from the semantic representation."""

    def __init__(self, input_dim: int, num_modules: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_modules)),
        )

    def forward(self, z_sem: torch.Tensor) -> dict:
        module_prior_logits = self.net(z_sem)
        return {'module_prior_logits': module_prior_logits}


def grouped_softmax(fine_logits: torch.Tensor, gene_to_module: torch.Tensor, num_modules: int) -> torch.Tensor:
    """Apply softmax independently within each gene module."""
    if fine_logits.dim() != 2:
        raise ValueError(f"fine_logits must be (B,G), got {tuple(fine_logits.shape)}")
    bsz, num_genes = fine_logits.shape
    probs = torch.zeros_like(fine_logits)
    for m in range(int(num_modules)):
        idx = (gene_to_module == m).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        logits_m = fine_logits.index_select(1, idx)
        probs_m = torch.softmax(logits_m, dim=-1).to(dtype=probs.dtype)
        probs.scatter_(1, idx.view(1, -1).expand(bsz, -1), probs_m)
    return probs


def recompose_gene_composition(
    module_comp: torch.Tensor,
    gene_within_module: torch.Tensor,
    gene_to_module: torch.Tensor,
) -> torch.Tensor:
    """Compute gene_comp[g] = module_comp[module(g)] * within[g]."""
    if module_comp.dim() != 2 or gene_within_module.dim() != 2:
        raise ValueError("module_comp/gene_within_module must be 2D")
    bsz, num_genes = gene_within_module.shape
    gather_idx = gene_to_module.view(1, num_genes).expand(bsz, -1)
    module_mass = torch.gather(module_comp, 1, gather_idx)
    gene_comp = module_mass * gene_within_module
    denom = gene_comp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return gene_comp / denom


class GMCPDecoder(nn.Module):
    """GMCP: coarse(module) + fine(within-module) -> gene composition."""

    def __init__(
        self,
        *,
        cond_dim: int,
        num_genes: int,
        num_modules: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        prior_alpha: float = 1.0,
    ):
        super().__init__()
        self.num_genes = int(num_genes)
        self.num_modules = int(num_modules)
        self.prior_alpha = float(prior_alpha)

        self.coarse_head = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_modules),
        )
        self.fine_head = nn.Sequential(
            nn.Linear(cond_dim + self.num_modules * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_genes),
        )

    def forward(
        self,
        z_final: torch.Tensor,
        module_prior_logits: torch.Tensor,
        gene_to_module: torch.Tensor,
    ) -> dict:
        coarse_logits = self.coarse_head(z_final) + self.prior_alpha * module_prior_logits
        module_comp = torch.softmax(coarse_logits, dim=-1)

        fine_in = torch.cat([z_final, module_comp, module_prior_logits], dim=-1)
        fine_logits = self.fine_head(fine_in)
        gene_within_module = grouped_softmax(
            fine_logits,
            gene_to_module=gene_to_module,
            num_modules=self.num_modules,
        )
        gene_comp = recompose_gene_composition(module_comp, gene_within_module, gene_to_module)

        return {
            'module_comp': module_comp,
            'gene_within_module': gene_within_module,
            'gene_comp': gene_comp,
            'coarse_logits': coarse_logits,
            'fine_logits': fine_logits,
        }
