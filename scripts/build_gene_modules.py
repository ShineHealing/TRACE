#!/usr/bin/env python3
"""Build training-only gene-module assignments."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import project_root, resolve_data_dir, resolve_path


def _normalize_sample_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _auto_train_samples(data_dir: Path, test_samples):
    all_samples = sorted([p.stem for p in data_dir.glob('*.h5')])
    return [s for s in all_samples if s not in set(test_samples)]


def _load_train_matrix(data_dir: Path, train_samples):
    mats = []
    for s in train_samples:
        p = data_dir / f'{s}.h5'
        if not p.exists():
            continue
        with h5py.File(p, 'r') as f:
            x = np.asarray(f['genes_count'], dtype=np.float32)
            mats.append(x)
    if not mats:
        raise RuntimeError('No training genes loaded.')
    return np.concatenate(mats, axis=0)


def _libsize_log1p_normalize(x: np.ndarray, scale: float = 1e4) -> np.ndarray:
    lib = np.clip(x.sum(axis=1, keepdims=True), 1e-8, None)
    x_norm = x / lib * float(scale)
    return np.log1p(x_norm)


def _kmeans_assign(gene_features: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=20)
        labels = km.fit_predict(gene_features)
        return labels.astype(np.int64)
    except Exception:
        rng = np.random.default_rng(int(seed))
        return rng.integers(0, int(k), size=(gene_features.shape[0],), endpoint=False, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description='Build fixed gene modules from training set.')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_samples', nargs='*', default=None)
    parser.add_argument('--test_samples', nargs='*', default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        config_path = resolve_path(args.config, base_dir=str(project_root()))
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    data_cfg = config.get('data', {}) or {}
    model_cfg = config.get('model', {}) or {}

    data_value = args.data_dir or data_cfg.get('data_dir')
    out_value = args.out_dir or data_cfg.get('gene_module_dir')
    if not data_value or not out_value:
        parser.error('--data_dir and --out_dir are required unless supplied by --config')
    data_dir = Path(resolve_data_dir(str(data_value)))
    out_dir = Path(resolve_path(str(out_value), base_dir=str(project_root())))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_value = args.train_samples or data_cfg.get('train_samples')
    train_mode = str(train_value).strip().upper() if isinstance(train_value, str) else None
    train_samples = (
        []
        if train_mode in ('AUTO', 'AUTO_EXCEPT_VAL_TEST')
        else _normalize_sample_list(train_value)
    )
    test_samples = _normalize_sample_list(args.test_samples or data_cfg.get('test_samples'))
    if not train_samples:
        train_samples = _auto_train_samples(data_dir, test_samples)
    if not train_samples:
        raise RuntimeError('Empty train_samples after split.')
    overlap = sorted(set(train_samples) & set(test_samples))
    if overlap:
        raise ValueError(f'Train/test leakage detected in gene-module construction: {overlap}')

    x = _load_train_matrix(data_dir, train_samples)
    x_norm = _libsize_log1p_normalize(x)

    
    gene_features = x_norm.T
    module_count = int(args.k or (model_cfg.get('decoder', {}) or {}).get('num_modules', 10))
    gene_to_module = _kmeans_assign(gene_features, k=module_count, seed=int(args.seed))
    module_sizes = np.bincount(gene_to_module, minlength=module_count).astype(np.int64)

    np.save(out_dir / 'gene_to_module.npy', gene_to_module)
    np.save(out_dir / 'module_sizes.npy', module_sizes)

    cfg = {
        'num_modules': module_count,
        'num_genes': int(gene_to_module.shape[0]),
        'seed': int(args.seed),
        'source_expression_field': 'genes_count',
        'normalization': 'libsize_1e4_log1p_for_clustering_only',
        'train_samples': train_samples,
    }
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f'[OK] Saved modules to: {out_dir}')
    print(f'    genes={cfg["num_genes"]}, modules={cfg["num_modules"]}, sizes={module_sizes.tolist()}')


if __name__ == '__main__':
    main()
