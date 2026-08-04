"""Training objective and fixed-budget trainer for TRACE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_true_composition(raw_counts: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    x = raw_counts.float().clamp_min(0.0)
    libsize = x.sum(dim=-1).clamp_min(eps)
    comp = x / libsize.unsqueeze(-1)
    return comp, libsize


def build_module_targets(
    gene_comp_true: torch.Tensor,
    gene_to_module: torch.Tensor,
    num_modules: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _num_genes = gene_comp_true.shape
    module_comp_true = torch.zeros(
        (batch_size, int(num_modules)),
        device=gene_comp_true.device,
        dtype=gene_comp_true.dtype,
    )
    gene_within = torch.zeros_like(gene_comp_true)

    for module_id in range(int(num_modules)):
        idx = (gene_to_module == module_id).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        comp_m = gene_comp_true.index_select(1, idx)
        mass_m = comp_m.sum(dim=-1, keepdim=True)
        module_comp_true[:, module_id] = mass_m.squeeze(-1)
        within_m = comp_m / mass_m.clamp_min(1e-8)
        gene_within.scatter_(1, idx.view(1, -1).expand(batch_size, -1), within_m)

    module_comp_true = module_comp_true / module_comp_true.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return module_comp_true, gene_within


def _edge_index_from_graph(graph: torch.Tensor | None, num_nodes: int) -> torch.Tensor | None:
    if graph is None or (not torch.is_tensor(graph)) or graph.numel() == 0:
        return None
    if graph.dim() == 2 and int(graph.shape[0]) == 2:
        edge_index = graph.long()
    elif graph.dim() == 2 and int(graph.shape[0]) == int(num_nodes) and int(graph.shape[1]) == int(num_nodes):
        edge_index = (graph > 0).nonzero(as_tuple=False).t().long()
    else:
        return None
    if edge_index.numel() == 0:
        return None
    src, dst = edge_index[0], edge_index[1]
    valid = (src >= 0) & (src < int(num_nodes)) & (dst >= 0) & (dst < int(num_nodes)) & (src != dst)
    if not bool(valid.all()):
        edge_index = edge_index[:, valid]
    return edge_index if edge_index.numel() > 0 else None


def graph_high_frequency_residual(u: torch.Tensor, graph: torch.Tensor | None) -> torch.Tensor:
    edge_index = _edge_index_from_graph(graph, int(u.shape[0]))
    if edge_index is None:
        return torch.zeros_like(u)

    src, dst = edge_index[0].to(u.device), edge_index[1].to(u.device)
    neigh_sum = torch.zeros_like(u)
    neigh_sum.index_add_(0, src, u.index_select(0, dst))

    degree = torch.zeros((u.shape[0],), device=u.device, dtype=u.dtype)
    degree.index_add_(0, src, torch.ones((src.numel(),), device=u.device, dtype=u.dtype))
    has_neighbor = degree > 0
    neigh_mean = u.clone()
    neigh_mean[has_neighbor] = neigh_sum[has_neighbor] / degree[has_neighbor].unsqueeze(-1).clamp_min(1.0)
    return u - neigh_mean


def compute_graph_hf_loss(pred_raw: torch.Tensor, true_raw: torch.Tensor, graph: torch.Tensor | None) -> torch.Tensor:
    edge_index = _edge_index_from_graph(graph, int(pred_raw.shape[0]))
    if edge_index is None:
        return pred_raw.new_zeros(())
    pred_u = torch.log1p(pred_raw.float().clamp_min(0.0))
    true_u = torch.log1p(true_raw.float().clamp_min(0.0))
    return F.smooth_l1_loss(
        graph_high_frequency_residual(pred_u, edge_index),
        graph_high_frequency_residual(true_u, edge_index),
    )


def compute_local_gradient_loss(pred_raw: torch.Tensor, true_raw: torch.Tensor, graph: torch.Tensor | None) -> torch.Tensor:
    edge_index = _edge_index_from_graph(graph, int(pred_raw.shape[0]))
    if edge_index is None:
        return pred_raw.new_zeros(())
    src, dst = edge_index[0].to(pred_raw.device), edge_index[1].to(pred_raw.device)
    pred_u = torch.log1p(pred_raw.float().clamp_min(0.0))
    true_u = torch.log1p(true_raw.float().clamp_min(0.0))
    pred_delta = pred_u.index_select(0, src) - pred_u.index_select(0, dst)
    true_delta = true_u.index_select(0, src) - true_u.index_select(0, dst)
    return F.smooth_l1_loss(pred_delta, true_delta)


def compute_final_framework_losses(
    outputs: dict,
    targets: dict,
    lambda_coarse: float = 0.05,
    lambda_scale: float = 0.05,
    lambda_log_mse: float = 0.0,
    lambda_gene_pcc: float = 0.0,
    lambda_hf: float = 0.0,
    lambda_grad: float = 0.0,
) -> dict:
    pred_raw = outputs.get('pred_raw_base', outputs['pred_raw']).float().clamp_min(0.0)
    pred_scale_raw = outputs['pred_scale'].float().clamp_min(0.0)
    pred_scale_log1p = outputs.get('pred_scale_log1p', torch.log1p(pred_scale_raw)).float().view(-1)
    pred_gene_comp = outputs['gene_comp'].float().clamp_min(1e-8)
    pred_module_comp = outputs['module_comp'].float().clamp_min(1e-8)

    true_raw = targets['raw_counts'].float().clamp_min(0.0)
    true_gene_comp = targets['gene_comp_true'].float().clamp_min(1e-8)
    true_module_comp = targets['module_comp_true'].float().clamp_min(1e-8)
    true_libsize = targets['libsize_true'].float().clamp_min(0.0)
    true_libsize_log1p = torch.log1p(true_libsize).view(-1)

    loss_final_raw = F.smooth_l1_loss(torch.log1p(pred_raw), torch.log1p(true_raw))
    loss_log_mse = F.mse_loss(torch.log1p(pred_raw), torch.log1p(true_raw))

    
    
    
    
    pred_log = torch.log1p(pred_raw)
    true_log = torch.log1p(true_raw)
    pred_centered = pred_log - pred_log.mean(dim=0, keepdim=True)
    true_centered = true_log - true_log.mean(dim=0, keepdim=True)
    pred_ss = pred_centered.square().sum(dim=0)
    true_ss = true_centered.square().sum(dim=0)
    valid_gene_pcc = true_ss > 1e-8
    if torch.any(valid_gene_pcc):
        gene_corr = (pred_centered * true_centered).sum(dim=0) / (
            torch.sqrt(pred_ss.clamp_min(1e-8)) * torch.sqrt(true_ss.clamp_min(1e-8))
        )
        loss_gene_pcc = 1.0 - gene_corr[valid_gene_pcc].mean()
    else:
        loss_gene_pcc = pred_raw.new_zeros(())
    loss_gene_composition = F.kl_div(torch.log(pred_gene_comp), true_gene_comp, reduction='batchmean')
    loss_module_coarse = F.kl_div(torch.log(pred_module_comp), true_module_comp, reduction='batchmean')
    loss_scale = F.smooth_l1_loss(pred_scale_log1p, true_libsize_log1p)

    graph = targets.get('edge_index', None)
    loss_hf = compute_graph_hf_loss(pred_raw, true_raw, graph)
    loss_grad = compute_local_gradient_loss(pred_raw, true_raw, graph)

    total = (
        loss_final_raw
        + float(lambda_log_mse) * loss_log_mse
        + float(lambda_gene_pcc) * loss_gene_pcc
        + loss_gene_composition
        + float(lambda_coarse) * loss_module_coarse
        + float(lambda_scale) * loss_scale
        + float(lambda_hf) * loss_hf
        + float(lambda_grad) * loss_grad
    )
    return {
        'loss': total,
        'loss_base': total,
        'loss_final_raw': loss_final_raw,
        'loss_log_mse': loss_log_mse,
        'loss_gene_pcc': loss_gene_pcc,
        'loss_gene_composition': loss_gene_composition,
        'loss_module_coarse': loss_module_coarse,
        'loss_scale': loss_scale,
        'loss_hf': loss_hf,
        'loss_grad': loss_grad,
    }


import os
import random
from typing import Dict, Optional

import numpy as np
import torch

from data import create_dataloaders
from models.trace import TRACE
from utils import project_root, resolve_data_dir, resolve_path, setup_logging


def _sample_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _list_h5_samples(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(data_dir)
        if name.endswith(".h5")
    )


class Trainer:
    """Trainer for the final TRACE model and fixed-budget experiments."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logging(config.get("logging", {}))
        training_cfg = config["training"]
        data_cfg = config["data"]

        self.seed = int(training_cfg.get("seed", 43))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._configure_runtime(config.get("performance", {}))
        self.logger.info("device=%s seed=%d", self.device, self.seed)

        data_dir = resolve_data_dir(data_cfg["data_dir"])
        test_samples = _sample_list(data_cfg.get("test_samples"))
        train_cfg = data_cfg.get("train_samples", "AUTO_EXCEPT_VAL_TEST")
        if isinstance(train_cfg, str) and train_cfg.upper() in {"AUTO", "AUTO_EXCEPT_VAL_TEST"}:
            excluded = set(test_samples)
            train_samples = [name for name in _list_h5_samples(data_dir) if name not in excluded]
        else:
            train_samples = _sample_list(train_cfg)
        if not train_samples:
            raise ValueError("The training split is empty.")
        overlap = sorted(set(train_samples) & set(test_samples))
        if overlap:
            raise ValueError(f"Train/test leakage detected: {overlap}")

        gene_module_dir = resolve_path(
            str(data_cfg["gene_module_dir"]), base_dir=str(project_root())
        )
        self.train_loader = create_dataloaders(
            data_dir=data_dir,
            train_samples=train_samples,
            batch_size=int(data_cfg.get("batch_size", 32)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            k_neighbors=int(data_cfg.get("k_neighbors", 10)),
            spatial_batching=bool(data_cfg.get("spatial_batching", True)),
            lazy_load=bool(data_cfg.get("lazy_load", False)),
            gene_module_dir=gene_module_dir,
            spatial_pack_size=int(data_cfg.get("spatial_pack_size", 1)),
        )

        dataset = self.train_loader.dataset
        expected_backbone = str(data_cfg.get('feature_backbone', '')).lower().strip()
        if expected_backbone not in {'resnet50', 'uni', 'uni2-h'}:
            raise ValueError(
                'data.feature_backbone must be one of: resnet50, uni, uni2-h'
            )
        if dataset.feature_backbone and dataset.feature_backbone != expected_backbone:
            raise ValueError(
                f'Processed features use {dataset.feature_backbone}, but the configuration '
                f'expects {expected_backbone}'
            )
        self.logger.info(
            'feature_backbone=%s feature_dim=%d',
            dataset.feature_backbone or expected_backbone,
            dataset.feature_dim,
        )
        if dataset.gene_to_module is None or dataset.num_modules <= 0:
            raise ValueError("gene_module_dir must contain gene_to_module.npy and module_sizes.npy")
        self.gene_to_module = np.asarray(dataset.gene_to_module, dtype=np.int64)
        self.num_modules = int(dataset.num_modules)

        model_cfg = dict(config["model"])
        if str(model_cfg.get("version", "")).lower() != "trace":
            raise ValueError("Only model.version=trace is supported.")
        model_cfg.setdefault("fdt", {})["input_dim"] = int(dataset.feature_dim)
        model_cfg.setdefault("decoder", {})["output_dim"] = int(dataset.num_genes)
        model_cfg["num_genes"] = int(dataset.num_genes)
        model_cfg["num_modules"] = self.num_modules
        self.model = TRACE(model_cfg).to(self.device)
        self.model.set_gene_to_module(
            torch.as_tensor(self.gene_to_module, dtype=torch.long, device=self.device)
        )

        self.num_epochs = int(training_cfg.get("num_epochs", 35))
        self.grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
        self.gradient_clip = float(training_cfg.get("gradient_clip", 1.0))
        self.loss_weights = {
            "lambda_coarse": float(training_cfg.get("lambda_coarse", 0.05)),
            "lambda_scale": float(training_cfg.get("lambda_scale", 0.2)),
            "lambda_log_mse": float(training_cfg.get("lambda_log_mse", 0.0)),
            "lambda_gene_pcc": float(training_cfg.get("lambda_gene_pcc", 0.0)),
            "lambda_hf": float(training_cfg.get("lambda_hf", 0.0)),
            "lambda_grad": float(training_cfg.get("lambda_grad", 0.0)),
        }
        self.hf_loss_start_epoch = max(0, int(training_cfg.get("hf_loss_start_epoch", 0)))
        self.hf_temperature_schedule = training_cfg.get("hf_alpha_temperature_schedule") or []
        self.hf_alpha_usage_lambda = float(training_cfg.get("hf_alpha_usage_lambda", 0.0))
        self.hf_alpha_usage_prior = training_cfg.get("hf_alpha_usage_prior")

        amp_name = str(training_cfg.get("amp_dtype", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_name in {"bf16", "bfloat16"} else torch.float16
        self.amp_enabled = bool(training_cfg.get("mixed_precision", False) and self.device.type == "cuda")
        self.use_scaler = bool(self.amp_enabled and self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(training_cfg.get("scheduler_t_max", self.num_epochs)),
            eta_min=float(training_cfg.get("scheduler_eta_min", 1e-6)),
        )

        checkpoint_cfg = training_cfg.get("checkpoint", {}) or {}
        self.checkpoint_dir = resolve_path(
            str(checkpoint_cfg.get("save_dir", "checkpoints")), base_dir=str(project_root())
        )
        self.save_last = bool(checkpoint_cfg.get("save_last", True))
        self.save_every = int(checkpoint_cfg.get("save_every_n_epochs", 0))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = 0
        self.last_epoch = -1

    def _configure_runtime(self, performance_cfg: dict) -> None:
        if self.device.type != "cuda":
            return
        torch.backends.cuda.matmul.allow_tf32 = bool(performance_cfg.get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(performance_cfg.get("tf32", True))
        torch.backends.cudnn.benchmark = bool(performance_cfg.get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision(str(performance_cfg.get("matmul_precision", "high")))

    def _set_hf_temperature(self, epoch: int) -> None:
        if not self.hf_temperature_schedule:
            return
        temperature = None
        for until_epoch, value in self.hf_temperature_schedule:
            temperature = float(value)
            if epoch + 1 <= int(until_epoch):
                break
        if temperature is not None:
            self.model.fdt.set_hf_alpha_temperature(temperature)

    def _targets(self, raw_counts: torch.Tensor, edge_index: Optional[torch.Tensor]) -> dict:
        gene_comp, libsize = compute_true_composition(raw_counts)
        gene_to_module = torch.as_tensor(
            self.gene_to_module, dtype=torch.long, device=raw_counts.device
        )
        module_comp, gene_within_module = build_module_targets(
            gene_comp, gene_to_module, self.num_modules
        )
        return {
            "raw_counts": raw_counts.float().clamp_min(0.0),
            "gene_comp_true": gene_comp,
            "module_comp_true": module_comp,
            "gene_within_module_true": gene_within_module,
            "libsize_true": libsize,
            "edge_index": edge_index,
        }

    def _loss(self, outputs: dict, targets: dict, epoch: int) -> dict:
        weights = dict(self.loss_weights)
        if epoch < self.hf_loss_start_epoch:
            weights["lambda_hf"] = 0.0
            weights["lambda_grad"] = 0.0
        losses = compute_final_framework_losses(outputs, targets, **weights)

        alpha = outputs.get("hf_alpha_loc")
        if self.hf_alpha_usage_lambda > 0 and torch.is_tensor(alpha) and alpha.shape[1] > 1:
            mean_alpha = alpha.float().mean(dim=0).clamp_min(1e-6)
            mean_alpha = mean_alpha / mean_alpha.sum()
            if self.hf_alpha_usage_prior is None:
                prior = torch.full_like(mean_alpha, 1.0 / mean_alpha.numel())
            else:
                prior = torch.as_tensor(
                    self.hf_alpha_usage_prior, dtype=mean_alpha.dtype, device=mean_alpha.device
                )
                if prior.numel() != mean_alpha.numel():
                    raise ValueError("hf_alpha_usage_prior does not match the number of HF bands")
                prior = prior.clamp_min(1e-6)
                prior = prior / prior.sum()
            usage_loss = torch.sum(prior * (torch.log(prior) - torch.log(mean_alpha)))
            losses["loss"] = losses["loss"] + self.hf_alpha_usage_lambda * usage_loss
            losses["loss_alpha_usage"] = usage_loss
        return losses

    def _optimizer_step(self) -> bool:
        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in self.model.parameters()
        )
        if not finite:
            self.optimizer.zero_grad(set_to_none=True)
            self.logger.warning("Skipping an optimizer step with non-finite gradients.")
            return False
        if self.use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        finite_batches = 0
        pending_steps = 0

        for batch in self.train_loader:
            local = batch["local_features"].to(self.device, non_blocking=True)
            macro = batch["macro_features"].to(self.device, non_blocking=True)
            coords = batch["coords_norm"].to(self.device, non_blocking=True)
            genes = batch["genes"].to(self.device, non_blocking=True)
            edge_index = batch.get("edge_index")
            if edge_index is not None:
                edge_index = edge_index.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
            ):
                outputs = self.model(local, macro, coords, edge_index)
                losses = self._loss(outputs, self._targets(genes, edge_index), epoch)
                loss = losses["loss"]
            if not bool(torch.isfinite(loss)):
                self.optimizer.zero_grad(set_to_none=True)
                pending_steps = 0
                self.logger.warning("Skipping a batch with non-finite loss.")
                continue

            scaled_loss = loss / self.grad_accum_steps
            if self.use_scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_steps += 1
            if pending_steps == self.grad_accum_steps:
                self._optimizer_step()
                pending_steps = 0

            finite_batches += 1
            for name, value in losses.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value.detach())

        if pending_steps:
            self._optimizer_step()
        if finite_batches == 0:
            raise RuntimeError("No finite training batches were produced.")
        return {name: value / finite_batches for name, value in totals.items()}

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            self._set_hf_temperature(epoch)
            metrics = self.train_epoch(epoch)
            self.scheduler.step()
            self.last_epoch = epoch
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                "epoch=%d/%d loss=%.6f raw=%.6f comp=%.6f scale=%.6f lr=%.3e",
                epoch + 1,
                self.num_epochs,
                metrics["loss"],
                metrics["loss_final_raw"],
                metrics["loss_gene_composition"],
                metrics["loss_scale"],
                lr,
            )
            if self.save_last:
                self.save_model(os.path.join(self.checkpoint_dir, "last_model.pth"), epoch)
            if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                self.save_model(
                    os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}.pth"), epoch
                )

    def save_model(self, path: str, epoch: Optional[int] = None) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.config,
                "epoch": self.last_epoch if epoch is None else int(epoch),
            },
            path,
        )
        self.logger.info("checkpoint=%s", path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True, mmap=True)
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            self.logger.warning(
                "Checkpoint compatibility: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
        if isinstance(checkpoint, dict):
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.last_epoch = self.start_epoch - 1
        self.logger.info("resumed=%s start_epoch=%d", path, self.start_epoch)


__all__ = [
    "Trainer",
    "compute_true_composition",
    "build_module_targets",
    "compute_final_framework_losses",
]
