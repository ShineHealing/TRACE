"""Inference for TRACE using precomputed frozen image features."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import SpatialTranscriptomicsDataset, make_basic_collate_fn, make_spatial_collate_fn
from models.trace import TRACE
from utils import project_root, resolve_data_dir, resolve_path


def _load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _resolve_model_path(model_path: str, config: dict) -> str:
    if model_path != 'auto:last':
        return resolve_path(model_path, base_dir=str(project_root()))
    ckpt_cfg = ((config.get('training', {}) or {}).get('checkpoint', {}) or {})
    save_dir = ckpt_cfg.get('save_dir', 'checkpoints/')
    return os.path.join(resolve_path(str(save_dir), base_dir=str(project_root())), 'last_model.pth')


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((am * am).sum(axis=1) * (bm * bm).sum(axis=1))
    out = np.full((a.shape[0],), np.nan, dtype=np.float64)
    mask = denom > 1e-12
    out[mask] = (am[mask] * bm[mask]).sum(axis=1) / denom[mask]
    return out


def _nanmean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float('nan')


def _metrics(pred_raw: np.ndarray, target_raw: np.ndarray) -> dict[str, float]:
    pred = np.log1p(np.clip(pred_raw, 0, None))
    tgt = np.log1p(np.clip(target_raw, 0, None))
    return {
        'PCC_Gene_log1p': _nanmean(_pearson_rows(pred.T, tgt.T)),
        'PCC_Spot_log1p': _nanmean(_pearson_rows(pred, tgt)),
        'MSE_log1p': float(np.mean((pred - tgt) ** 2)),
        'MAE_log1p': float(np.mean(np.abs(pred - tgt))),
    }


def _load_gene_names(data_dir: str, sample: str) -> np.ndarray | None:
    path = os.path.join(data_dir, f'{sample}.h5')
    if not os.path.exists(path):
        return None
    with h5py.File(path, 'r') as f:
        if 'gene_names' not in f:
            return None
        return np.asarray(f['gene_names'][:])


def _load_spot_ids(data_dir: str, sample: str) -> np.ndarray | None:
    path = os.path.join(data_dir, f'{sample}.h5')
    if not os.path.exists(path):
        return None
    with h5py.File(path, 'r') as f:
        if 'barcodes' not in f:
            return None
        return np.asarray(f['barcodes'][:])


def _load_feature_attrs(data_dir: str, sample: str) -> dict:
    path = os.path.join(data_dir, f'{sample}.h5')
    if not os.path.exists(path):
        return {}
    keys = ('feature_backbone', 'feature_dim', 'feature_view', 'local_patch_size', 'macro_patch_size')
    out = {}
    with h5py.File(path, 'r') as f:
        for k in keys:
            if k in f.attrs:
                v = f.attrs[k]
                out[k] = v.decode('utf-8') if isinstance(v, bytes) else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Run TRACE inference on one processed sample.')
    ap.add_argument('--config', default='src/configs/example.yaml')
    ap.add_argument('--model_path', default='auto:last')
    ap.add_argument('--test_sample', default=None)
    ap.add_argument('--output_file', default='results/predictions_her2st_last.h5')
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument(
        '--save-target-for-eval',
        action='store_true',
        help='Write target_raw and evaluation metrics into the output H5. Default output is public-safe.',
    )
    args = ap.parse_args()

    config_path = resolve_path(args.config, base_dir=str(project_root()))
    config = _load_config(config_path)
    data_cfg = config['data']
    model_cfg = dict(config['model'])
    model_version = str(model_cfg.get('version', '')).lower().strip()
    if model_version != 'trace':
        raise ValueError('Inference supports only model.version=trace')

    data_dir = resolve_data_dir(data_cfg['data_dir'])
    test_sample = args.test_sample or (data_cfg.get('test_samples') or [None])[0]
    if not test_sample:
        raise ValueError('No test sample provided and data.test_samples is empty')

    gene_module_dir = data_cfg.get('gene_module_dir', None)
    if gene_module_dir:
        gene_module_dir = resolve_path(str(gene_module_dir), base_dir=str(project_root()))

    dataset = SpatialTranscriptomicsDataset(
        data_dir=data_dir,
        sample_names=[test_sample],
        lazy_load=True,
        gene_module_dir=gene_module_dir,
    )
    expected_backbone = str(data_cfg.get('feature_backbone', '')).lower().strip()
    if expected_backbone not in {'resnet50', 'uni', 'uni2-h'}:
        raise ValueError('data.feature_backbone must be one of: resnet50, uni, uni2-h')
    if dataset.feature_backbone and dataset.feature_backbone != expected_backbone:
        raise ValueError(
            f'Processed features use {dataset.feature_backbone}, but the configuration '
            f'expects {expected_backbone}'
        )

    model_cfg.setdefault('fdt', {})
    model_cfg.setdefault('decoder', {})
    model_cfg['fdt']['input_dim'] = int(dataset.feature_dim)
    model_cfg['decoder']['output_dim'] = int(dataset.num_genes)
    model_cfg['decoder']['num_modules'] = int(dataset.num_modules)
    model_cfg['feature_dim'] = int(dataset.feature_dim)
    model_cfg['input_dim'] = int(dataset.feature_dim)
    model_cfg['output_dim'] = int(dataset.num_genes)
    model_cfg['num_genes'] = int(dataset.num_genes)
    model_cfg['num_modules'] = int(dataset.num_modules)
    model_cfg['prefix_len'] = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TRACE(model_cfg).to(device)
    if dataset.gene_to_module is None:
        raise ValueError('gene_module_dir is required for TRACE inference')
    model.set_gene_to_module(torch.as_tensor(dataset.gene_to_module, dtype=torch.long, device=device))

    ckpt_path = _resolve_model_path(args.model_path, config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True, mmap=True)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    except TypeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[CheckpointLoad] missing={len(missing)} unexpected={len(unexpected)}')
    model.eval()

    batch_size = int(args.batch_size or data_cfg.get('val_batch_size') or data_cfg.get('batch_size', 64))
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get('num_workers', 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=(
            make_spatial_collate_fn(k_neighbors=int(data_cfg.get('k_neighbors', 10)))
            if bool(data_cfg.get('spatial_batching', True))
            else make_basic_collate_fn()
        ),
    )

    preds, pred_scales, gene_comps, targets, coords = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'infer {test_sample}'):
            local = batch['local_features'].to(device, non_blocking=True)
            macro = batch['macro_features'].to(device, non_blocking=True)
            xy = batch['coords_norm'].to(device, non_blocking=True)
            edge_index = batch.get('edge_index', None)
            if edge_index is not None:
                edge_index = edge_index.to(device, non_blocking=True)
            out = model(local, macro, xy, edge_index)
            preds.append(out['pred_raw'].detach().cpu().numpy())
            if 'pred_scale' in out:
                pred_scales.append(out['pred_scale'].detach().cpu().numpy())
            if 'gene_comp' in out:
                gene_comps.append(out['gene_comp'].detach().cpu().numpy())
            if args.save_target_for_eval:
                targets.append(batch['genes'].detach().cpu().numpy())
            coords.append(batch['coords'].detach().cpu().numpy())

    pred_raw = np.concatenate(preds, axis=0)
    pred_scale_raw = np.concatenate(pred_scales, axis=0) if pred_scales else None
    gene_comp = np.concatenate(gene_comps, axis=0) if gene_comps else None
    coords_np = np.concatenate(coords, axis=0)
    target_raw = np.concatenate(targets, axis=0) if args.save_target_for_eval else None
    metrics = _metrics(pred_raw, target_raw) if target_raw is not None else {}

    out_path = resolve_path(args.output_file, base_dir=str(project_root()))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    gene_names = _load_gene_names(data_dir, test_sample)
    spot_ids = _load_spot_ids(data_dir, test_sample)
    feature_attrs = _load_feature_attrs(data_dir, test_sample)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('pred_raw', data=pred_raw.astype(np.float32), compression='gzip')
        if pred_scale_raw is not None:
            f.create_dataset('pred_scale_raw', data=pred_scale_raw.astype(np.float32), compression='gzip')
        if gene_comp is not None:
            f.create_dataset('gene_comp', data=gene_comp.astype(np.float32), compression='gzip')
        f.create_dataset('coords', data=coords_np.astype(np.float32), compression='gzip')
        if target_raw is not None:
            f.create_dataset('target_raw', data=target_raw.astype(np.float32), compression='gzip')
        if spot_ids is not None:
            f.create_dataset('spot_ids', data=spot_ids)
        if gene_names is not None:
            f.create_dataset('gene_names', data=gene_names)
        for k, v in metrics.items():
            f.attrs[k] = float(v)
        f.attrs['sample'] = str(test_sample)
        f.attrs['checkpoint'] = str(ckpt_path)
        f.attrs['contains_target_raw'] = bool(target_raw is not None)
        f.attrs['feature_backbone'] = str(dataset.feature_backbone or expected_backbone)
        f.attrs['model_version'] = str(model_version)
        for k, v in feature_attrs.items():
            f.attrs[k] = v

    print(f'output: {out_path}')
    if metrics:
        for k, v in metrics.items():
            print(f'{k}: {v:.6f}')
    else:
        print('target_raw: not saved (use --save-target-for-eval for internal evaluation files)')


if __name__ == '__main__':
    main()
