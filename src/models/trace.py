import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Optional

from models.fdt import FDT
from models.gmcp import GMCPDecoder, MPN


class TRACE(nn.Module):
    """FDT + GMCP + composition-scale factorization."""

    def __init__(self, config: Dict):
        super().__init__()
        fdt_cfg = (config.get('fdt', {}) or {})
        decoder_cfg = (config.get('decoder', {}) or {})
        mpn_cfg = (config.get('mpn', {}) or {})
        scale_cfg = (config.get('scale_head', {}) or {})

        feature_dim = int(fdt_cfg.get('input_dim', 1024))
        hidden_dim = int(fdt_cfg.get('hidden_dim', 512))
        cond_dim = int(fdt_cfg.get('output_dim', 512))
        num_genes = int(decoder_cfg.get('output_dim', config.get('num_genes', 200)))
        num_modules = int(config.get('num_modules', decoder_cfg.get('num_modules', 10)))

        self.num_genes = num_genes
        self.num_modules = num_modules

        self.fdt = FDT(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=cond_dim,
            num_frequencies=int(fdt_cfg.get('num_frequencies', 64)),
            num_graph_layers=int(fdt_cfg.get('num_graph_layers', 3)),
            dropout=float(fdt_cfg.get('dropout', 0.1)),
            aggregator_type=str(fdt_cfg.get('aggregator_type', 'gat')),
            num_heads=int(fdt_cfg.get('num_heads', 8)),
            hf_lambda=float(fdt_cfg.get('hf_lambda', 0.25)),
            hf_use_local_feat=bool(fdt_cfg.get('hf_use_local_feat', True)),
            hf_residual_type=str(fdt_cfg.get('hf_residual_type', 'pos_mlp')),
            hf_residual_scale=float(fdt_cfg.get('hf_residual_scale', 1.0)),
            hf_alpha_temperature=float(fdt_cfg.get('hf_alpha_temperature', 1.0)),
            hf_gamma_max=float(fdt_cfg.get('hf_gamma_max', 0.3)),
            hf_gamma_init=float(fdt_cfg.get('hf_gamma_init', 0.1)),
        )
        self.fdt.hf_num_frequency_orders = int(fdt_cfg.get('hf_num_frequency_orders', 3))

        self.mpn = MPN(
            input_dim=cond_dim,
            num_modules=num_modules,
            hidden_dim=int(mpn_cfg.get('hidden_dim', 256)),
            dropout=float(mpn_cfg.get('dropout', 0.1)),
        )

        self.gmcp = GMCPDecoder(
            cond_dim=cond_dim,
            num_genes=num_genes,
            num_modules=num_modules,
            hidden_dim=int(decoder_cfg.get('hidden_dim', 512)),
            dropout=float(decoder_cfg.get('dropout', 0.1)),
            prior_alpha=float(decoder_cfg.get('prior_alpha', 1.0)),
        )

        scale_hidden = int(scale_cfg.get('hidden_dim', 256))
        scale_dropout = float(scale_cfg.get('dropout', 0.1))
        self.scale_head_input_source = str(scale_cfg.get('input_source', 'z_final') or 'z_final').lower().strip()
        if self.scale_head_input_source not in ('z_final', 'z_sem_local_macro'):
            raise ValueError(
                "scale_head.input_source must be 'z_final' or 'z_sem_local_macro'"
            )
        scale_input_dim = cond_dim
        self.scale_local_proj = None
        self.scale_macro_proj = None
        if self.scale_head_input_source == 'z_sem_local_macro':
            scale_feat_dim = int(scale_cfg.get('feature_proj_dim', 128))
            self.scale_local_proj = nn.Sequential(
                nn.Linear(feature_dim, scale_feat_dim),
                nn.LayerNorm(scale_feat_dim),
                nn.GELU(),
                nn.Dropout(scale_dropout),
            )
            self.scale_macro_proj = nn.Sequential(
                nn.Linear(feature_dim, scale_feat_dim),
                nn.LayerNorm(scale_feat_dim),
                nn.GELU(),
                nn.Dropout(scale_dropout),
            )
            scale_input_dim = cond_dim + 2 * scale_feat_dim
        self.scale_head = nn.Sequential(
            nn.Linear(scale_input_dim, scale_hidden),
            nn.GELU(),
            nn.Dropout(scale_dropout),
            nn.Linear(scale_hidden, 1),
        )

        self.register_buffer('gene_to_module', torch.zeros((num_genes,), dtype=torch.long), persistent=False)

    def set_gene_to_module(self, gene_to_module: torch.Tensor) -> None:
        if gene_to_module.dim() != 1 or int(gene_to_module.numel()) != int(self.num_genes):
            raise ValueError(f'gene_to_module shape mismatch: expected ({self.num_genes},), got {tuple(gene_to_module.shape)}')
        self.register_buffer('gene_to_module', gene_to_module.long().detach().clone(), persistent=False)

    @staticmethod
    def _upgrade_legacy_state_dict(state_dict: dict) -> dict:
        """Translate pre-FDT checkpoint keys without changing current model names."""
        legacy_prefix = 'ptp.'
        current_prefix = 'fdt.'
        if not any(str(key).startswith(legacy_prefix) for key in state_dict):
            return state_dict
        upgraded = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if not str(key).startswith(legacy_prefix)
        )
        for key, value in state_dict.items():
            if str(key).startswith(legacy_prefix):
                new_key = current_prefix + str(key)[len(legacy_prefix):]
                upgraded.setdefault(new_key, value)
        if hasattr(state_dict, '_metadata'):
            upgraded._metadata = state_dict._metadata
        return upgraded

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        state_dict = self._upgrade_legacy_state_dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        local_feat: torch.Tensor,
        macro_feat: torch.Tensor,
        coords: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> dict:
        fdt_out = self.fdt(local_feat, macro_feat, coords, adj)
        z_sem = fdt_out['z_sem']
        z_final = fdt_out['z_final']

        mpn_out = self.mpn(z_sem)
        module_prior_logits = mpn_out['module_prior_logits']
        gmcp_out = self.gmcp(z_final, module_prior_logits, self.gene_to_module)

        
        if self.scale_head_input_source == 'z_sem_local_macro':
            scale_input = torch.cat(
                [
                    z_sem,
                    self.scale_local_proj(local_feat),
                    self.scale_macro_proj(macro_feat),
                ],
                dim=-1,
            )
        else:
            scale_input = z_final
        pred_scale_log1p = self.scale_head(scale_input).view(-1).clamp(min=-20.0, max=20.0)
        pred_scale_raw = torch.expm1(pred_scale_log1p).clamp_min(0.0)
        pred_raw_base = gmcp_out['gene_comp'] * pred_scale_raw.unsqueeze(-1)
        return {
            'pred_raw': pred_raw_base,
            'pred_scale': pred_scale_raw,
            'pred_scale_log1p': pred_scale_log1p,
            'gene_comp': gmcp_out['gene_comp'],
            'module_comp': gmcp_out['module_comp'],
            'gene_within_module': gmcp_out['gene_within_module'],
            'z_sem': z_sem,
            'z_final': z_final,
            'r_hf': fdt_out['r_hf'],
            'hf_alpha_loc': fdt_out.get('hf_alpha_loc'),
        }
