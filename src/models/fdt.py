"""FDT: frequency-decoupled topology-aware trunk and graph layers."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    """Graph attention layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        
        
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))
        
        if bias:
            if concat:
                self.bias = nn.Parameter(torch.empty(num_heads * out_features))
            else:
                self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.a_src, gain=gain)
        nn.init.xavier_uniform_(self.a_dst, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        N = x.size(0)
        device = x.device
        
        
        h = torch.einsum('ni,hio->hno', x, self.W)  
        
        
        
        attn_src = torch.einsum('hno,hod->hnd', h, self.a_src)  
        attn_dst = torch.einsum('hno,hod->hnd', h, self.a_dst)  
        
        
        if adj.dim() == 2 and adj.size(0) == N and adj.size(1) == N:
            
            
            adj_with_self = adj + torch.eye(N, device=device, dtype=adj.dtype)
            
            
            
            attn = attn_src + attn_dst.transpose(-2, -1)
            attn = F.leaky_relu(attn.squeeze(-1), negative_slope=self.negative_slope)
            
            
            mask = adj_with_self == 0
            attn = attn.masked_fill(mask.unsqueeze(0), float('-inf'))
            
            
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            
            
            out = torch.bmm(attn, h)
            
        else:
            
            row, col = adj[0], adj[1]
            
            
            loop = torch.arange(N, device=device)
            row = torch.cat([row, loop])
            col = torch.cat([col, loop])
            E = row.size(0)
            
            
            attn_edge = attn_src[:, row, :] + attn_dst[:, col, :]  
            attn_edge = F.leaky_relu(attn_edge.squeeze(-1), negative_slope=self.negative_slope)
            
            
            attn_edge = self._sparse_softmax(attn_edge, col, N)  
            attn_edge = F.dropout(attn_edge, p=self.dropout, training=self.training)
            
            
            out = self._scatter_add(h[:, col] * attn_edge.unsqueeze(-1), row, N)  
        
        
        if self.concat:
            out = out.transpose(0, 1).contiguous().view(N, -1)  
        else:
            out = out.mean(dim=0)  
        
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _sparse_softmax(self, attn: torch.Tensor, index: torch.Tensor, N: int) -> torch.Tensor:
        H = attn.size(0)
        attn_max = torch.zeros(H, N, device=attn.device, dtype=attn.dtype).scatter_reduce(
            1, index.unsqueeze(0).expand(H, -1), attn, reduce='amax', include_self=False
        )
        attn = attn - attn_max[:, index]
        attn_exp = attn.exp()
        attn_sum = torch.zeros(H, N, device=attn.device, dtype=attn.dtype).scatter_add(
            1, index.unsqueeze(0).expand(H, -1), attn_exp
        )
        return attn_exp / (attn_sum[:, index] + 1e-12)
    
    def _scatter_add(self, src: torch.Tensor, index: torch.Tensor, N: int) -> torch.Tensor:
        H, E, D = src.shape
        out = torch.zeros(H, N, D, device=src.device, dtype=src.dtype)
        index_expanded = index.unsqueeze(0).unsqueeze(-1).expand(H, -1, D)
        out.scatter_add_(1, index_expanded, src)
        return out


class GraphTransformerLayer(nn.Module):
    """Graph transformer layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_features: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    @staticmethod
    def _attention_mask_from_adj(adj: torch.Tensor, num_nodes: int, device: torch.device) -> Optional[torch.Tensor]:
        if adj is None or adj.numel() == 0:
            return None

        if adj.dim() == 2 and int(adj.shape[0]) == int(num_nodes) and int(adj.shape[1]) == int(num_nodes):
            allowed = adj.to(device=device) != 0
        elif adj.dim() == 2 and int(adj.shape[0]) == 2:
            edge_index = adj.to(device=device, dtype=torch.long)
            src, dst = edge_index[0], edge_index[1]
            valid = (src >= 0) & (src < int(num_nodes)) & (dst >= 0) & (dst < int(num_nodes))
            allowed = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
            if bool(valid.any()):
                allowed[src[valid], dst[valid]] = True
        else:
            return None

        loop = torch.arange(num_nodes, device=device)
        allowed[loop, loop] = True
        return ~allowed
    
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, D = x.shape
        device = x.device
        
        
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        
        attn_mask = self._attention_mask_from_adj(adj, N, device) if adj is not None else None
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = residual + out
        
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x


class HierarchicalGraphEncoder(nn.Module):
    """Hierarchical graph encoder."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_gat_layers: int = 2,
        num_transformer_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    num_heads=num_heads,
                    concat=True,
                    dropout=dropout,
                )
            )
        
        
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        
        for gat in self.gat_layers:
            h_new = gat(h, adj) if adj is not None else h
            h = F.elu(h_new) + h  
            h = self.dropout(h)
        
        
        for transformer in self.transformer_layers:
            h = transformer(h, adj)
        
        
        out = self.output_proj(h)
        
        return out

class GaussianFourierPositionalEncoding(nn.Module):
    """Encode normalized slide coordinates with random Fourier features."""

    def __init__(self, coord_dim: int = 2, embedding_size: int = 128, scale: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(embedding_size, coord_dim) * scale)
        self.output_dim = 2 * embedding_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projected = 2 * math.pi * torch.matmul(coords, self.B.T)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class NeighborAggregator(nn.Module):
    """Spatial aggregation used by the final TRACE configurations."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregator_type: str = "gat",
        num_heads: int = 4,
    ):
        super().__init__()
        if aggregator_type not in {"gat", "hierarchical"}:
            raise ValueError("aggregator_type must be 'gat' or 'hierarchical'")
        self.aggregator_type = aggregator_type

        if aggregator_type == "hierarchical":
            self.encoder = HierarchicalGraphEncoder(
                in_features=feature_dim,
                hidden_dim=hidden_dim,
                out_features=hidden_dim,
                num_gat_layers=num_layers,
                num_transformer_layers=1,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.input_proj = nn.Linear(feature_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        num_heads=num_heads,
                        concat=True,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fallback_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

    def forward(self, features: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_index is None:
            return self.fallback_proj(features)
        if self.aggregator_type == "hierarchical":
            return self.encoder(features, edge_index)
        hidden = self.input_proj(features)
        for layer, norm in zip(self.layers, self.norms):
            hidden = self.dropout(norm(F.elu(layer(hidden, edge_index)) + hidden))
        return hidden


class FDT(nn.Module):
    """Frequency-decoupled topology-aware trunk used by TRACE."""

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_frequencies: int = 64,
        num_graph_layers: int = 3,
        dropout: float = 0.1,
        aggregator_type: str = "gat",
        num_heads: int = 8,
        hf_lambda: float = 0.25,
        hf_use_local_feat: bool = True,
        hf_residual_type: str = "pos_mlp",
        hf_residual_scale: float = 1.0,
        hf_alpha_temperature: float = 1.0,
        hf_gamma_max: float = 0.3,
        hf_gamma_init: float = 0.1,
    ):
        super().__init__()
        if hf_residual_type not in {"pos_mlp", "local_bandpass"}:
            raise ValueError("hf_residual_type must be 'pos_mlp' or 'local_bandpass'")
        self.hf_lambda = float(hf_lambda)
        self.hf_use_local_feat = bool(hf_use_local_feat)
        self.hf_residual_type = hf_residual_type
        self.hf_residual_scale = float(hf_residual_scale)
        self.hf_num_frequency_orders = 3
        self.hf_gamma_max = float(hf_gamma_max)
        self.hf_gamma_init = float(hf_gamma_init)
        self.hf_alpha_temperature = max(float(hf_alpha_temperature), 1e-3)

        self.pos_encoder = GaussianFourierPositionalEncoding(
            coord_dim=2, embedding_size=num_frequencies * 2
        )
        aggregate_kwargs = dict(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_graph_layers,
            dropout=dropout,
            aggregator_type=aggregator_type,
            num_heads=max(1, num_heads // 2),
        )
        self.local_aggregator = NeighborAggregator(**aggregate_kwargs)
        self.macro_aggregator = NeighborAggregator(**aggregate_kwargs)

        semantic_dim = hidden_dim + self.pos_encoder.output_dim
        self.semantic_local = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.semantic_macro = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.pos_encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.z_proj = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))

        hf_input_dim = self.pos_encoder.output_dim + (feature_dim if hf_use_local_feat else 0)
        self.hf_mlp = nn.Sequential(
            nn.Linear(hf_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.hf_carrier_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim), nn.LayerNorm(output_dim)
        )
        self.hf_order_norms_loc = nn.ModuleList(
            [nn.LayerNorm(output_dim) for _ in range(3)]
        )
        self.hf_contrast_residual = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hf_contrast_gate = nn.Sequential(
            nn.Linear(2 * output_dim + self.pos_encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.hf_bandpass_alpha = nn.Sequential(
            nn.Linear(output_dim + hidden_dim + self.pos_encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.hf_gamma_head = nn.Sequential(
            nn.Linear(output_dim + self.pos_encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_hf_gamma_head()

    def set_hf_alpha_temperature(self, temperature: float) -> None:
        self.hf_alpha_temperature = max(float(temperature), 1e-3)

    def _init_hf_gamma_head(self) -> None:
        final_layer = self.hf_gamma_head[-1]
        nn.init.zeros_(final_layer.weight)
        ratio = min(
            max(self.hf_gamma_init / max(self.hf_gamma_max, 1e-6), 1e-6),
            1.0 - 1e-6,
        )
        nn.init.constant_(final_layer.bias, math.log(ratio / (1.0 - ratio)))

    @staticmethod
    def _neighbor_mean(features: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return features
        source, target = edge_index.long()
        node_count = features.shape[0]
        valid = (
            (source >= 0)
            & (source < node_count)
            & (target >= 0)
            & (target < node_count)
            & (source != target)
        )
        source, target = source[valid], target[valid]
        if source.numel() == 0:
            return features
        neighbor_sum = torch.zeros_like(features)
        neighbor_sum.index_add_(0, source, features.index_select(0, target))
        degree = torch.zeros(node_count, device=features.device, dtype=features.dtype)
        degree.index_add_(0, source, torch.ones_like(source, dtype=features.dtype))
        result = features.clone()
        mask = degree > 0
        result[mask] = neighbor_sum[mask] / degree[mask, None]
        return result

    def _local_bandpass(
        self,
        local_features: torch.Tensor,
        macro_context: torch.Tensor,
        z_sem: torch.Tensor,
        position: torch.Tensor,
        edge_index: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smoothed = self.hf_carrier_proj(local_features)
        bands = []
        order_count = max(1, min(self.hf_num_frequency_orders, len(self.hf_order_norms_loc)))
        for norm in list(self.hf_order_norms_loc)[:order_count]:
            next_smoothed = self._neighbor_mean(smoothed, edge_index)
            bands.append(norm(smoothed - next_smoothed))
            smoothed = next_smoothed
        bands = torch.stack(bands, dim=1)

        alpha_input = torch.cat([z_sem, macro_context, position], dim=-1)
        alpha_logits = self.hf_bandpass_alpha(alpha_input)[:, :order_count]
        alpha = torch.softmax(alpha_logits / self.hf_alpha_temperature, dim=-1)
        mixed_band = (alpha.unsqueeze(-1) * bands).sum(dim=1)
        gate = self.hf_contrast_gate(torch.cat([z_sem, mixed_band, position], dim=-1))
        residual = self.hf_residual_scale * gate * self.hf_contrast_residual(mixed_band)
        gamma = self.hf_gamma_max * torch.sigmoid(
            self.hf_gamma_head(torch.cat([z_sem, position], dim=-1))
        )
        return residual, gamma, alpha

    def forward(
        self,
        local_features: torch.Tensor,
        macro_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> dict:
        position = self.pos_encoder(coords)
        local_context = self.semantic_local(
            torch.cat([self.local_aggregator(local_features, edge_index), position], dim=-1)
        )
        macro_context = self.semantic_macro(
            torch.cat([self.macro_aggregator(macro_features, edge_index), position], dim=-1)
        )
        weights = torch.softmax(
            self.router(torch.cat([local_context, macro_context, position], dim=-1)), dim=-1
        )
        z_sem = self.z_proj(
            weights[:, :1] * local_context + weights[:, 1:] * macro_context
        )

        if self.hf_residual_type == "local_bandpass":
            residual, gamma, alpha = self._local_bandpass(
                local_features, macro_context, z_sem, position, edge_index
            )
            z_final = z_sem + gamma * residual
        else:
            hf_input = (
                torch.cat([position, local_features], dim=-1)
                if self.hf_use_local_feat
                else position
            )
            residual = self.hf_mlp(hf_input)
            z_final = z_sem + self.hf_lambda * residual
            alpha = None

        return {
            "z_sem": z_sem,
            "z_final": z_final,
            "r_hf": residual,
            "hf_alpha_loc": alpha,
        }
