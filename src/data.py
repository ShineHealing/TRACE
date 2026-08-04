from __future__ import annotations

import os
import threading
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None


class SpatialTranscriptomicsDataset(Dataset):
    """Processed spatial transcriptomics dataset."""

    def __init__(
        self,
        data_dir: str,
        sample_names: List[str],
        transform=None,
        target_transform=None,
        lazy_load: bool = False,
        gene_module_dir: Optional[str] = None,
    ):
        self.data_dir = str(data_dir)
        self.sample_names = [str(x) for x in (sample_names or [])]
        self.transform = transform
        self.target_transform = target_transform
        self.lazy_load = bool(lazy_load)
        self.gene_module_dir = gene_module_dir

        self._gene_to_module: Optional[np.ndarray] = None
        self._module_sizes: Optional[np.ndarray] = None
        self._num_modules = 0
        self._load_gene_modules(gene_module_dir)

        self.sample_indices: List[Tuple[int, int, str]] = []
        self.sample_name_by_id: List[str] = []
        self._h5_paths: List[str] = []
        self._sample_n_spots: List[int] = []
        self._edge_indices: List[np.ndarray] = []
        self._total_spots = 0
        self._num_genes = 0
        self._feature_dim = 0
        self._feature_backbone: Optional[str] = None

        self._h5_cache: Dict[str, h5py.File] = {}
        self._cache_lock = threading.Lock()

        self.local_features: Optional[np.ndarray] = None
        self.macro_features: Optional[np.ndarray] = None
        self.coords: Optional[np.ndarray] = None
        self.coords_norm: Optional[np.ndarray] = None
        self.genes: Optional[np.ndarray] = None
        self.sample_id: Optional[np.ndarray] = None
        self.local_idx: Optional[np.ndarray] = None

        if self.lazy_load:
            self._init_lazy()
        else:
            self._init_preload()

    def _load_gene_modules(self, gene_module_dir: Optional[str]) -> None:
        if not gene_module_dir:
            return
        g2m_path = os.path.join(str(gene_module_dir), 'gene_to_module.npy')
        sz_path = os.path.join(str(gene_module_dir), 'module_sizes.npy')
        if os.path.exists(g2m_path):
            self._gene_to_module = np.asarray(np.load(g2m_path), dtype=np.int64)
            self._num_modules = int(self._gene_to_module.max()) + 1 if self._gene_to_module.size > 0 else 0
        if os.path.exists(sz_path):
            self._module_sizes = np.asarray(np.load(sz_path), dtype=np.int64)
            if self._module_sizes.size > 0:
                self._num_modules = max(self._num_modules, int(self._module_sizes.shape[0]))

    def _sample_path(self, sample_name: str) -> str:
        return os.path.join(self.data_dir, f'{sample_name}.h5')

    @staticmethod
    def _expr_key(f: h5py.File) -> str:
        if 'genes_count' in f:
            return 'genes_count'
        raise KeyError('processed H5 must contain genes_count')

    @staticmethod
    def _read_coords_norm(f: h5py.File) -> np.ndarray:
        if 'coords_norm' in f:
            return np.asarray(f['coords_norm'][:], dtype=np.float32)
        coords = np.asarray(f['coords'][:], dtype=np.float32)
        cmin = coords.min(axis=0, keepdims=True)
        cmax = coords.max(axis=0, keepdims=True)
        return ((coords - cmin) / (cmax - cmin + 1e-8)).astype(np.float32)

    @staticmethod
    def _read_edge_index(f: h5py.File, n_spots: int) -> np.ndarray:
        if 'edge_index' not in f:
            raise KeyError('processed H5 must contain edge_index')
        edge = np.asarray(f['edge_index'][:], dtype=np.int64)
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f'edge_index must be (2,E), got {edge.shape}')
        if edge.size > 0 and (edge.min() < 0 or edge.max() >= int(n_spots)):
            raise ValueError('edge_index contains node ids outside [0, N)')
        return edge

    def _read_sample_metadata(
        self, f: h5py.File
    ) -> tuple[int, int, int, np.ndarray, np.ndarray, Optional[str]]:
        expr_key = self._expr_key(f)
        n_spots = int(f['local_features'].shape[0])
        feature_dim = int(f['local_features'].shape[1])
        num_genes = int(f[expr_key].shape[1])
        coords_norm = self._read_coords_norm(f)
        edge_index = self._read_edge_index(f, n_spots)
        backbone = f.attrs.get('feature_backbone')
        if isinstance(backbone, bytes):
            backbone = backbone.decode('utf-8')
        backbone = str(backbone).lower().strip() if backbone is not None else None
        return n_spots, feature_dim, num_genes, coords_norm, edge_index, backbone

    def _validate_sample_schema(
        self,
        *,
        sample_name: str,
        feature_dim: int,
        num_genes: int,
        feature_backbone: Optional[str],
    ) -> None:
        if self._feature_dim == 0:
            self._feature_dim = int(feature_dim)
            self._num_genes = int(num_genes)
        elif int(feature_dim) != self._feature_dim or int(num_genes) != self._num_genes:
            raise ValueError(
                f'Inconsistent H5 schema for {sample_name}: feature_dim={feature_dim}, '
                f'num_genes={num_genes}; expected feature_dim={self._feature_dim}, '
                f'num_genes={self._num_genes}'
            )
        if feature_backbone:
            if self._feature_backbone is None:
                self._feature_backbone = feature_backbone
            elif feature_backbone != self._feature_backbone:
                raise ValueError(
                    f'Inconsistent feature backbones: {sample_name} uses {feature_backbone}, '
                    f'expected {self._feature_backbone}'
                )

    def _register_sample(
        self,
        *,
        sample_idx: int,
        sample_name: str,
        n_spots: int,
        edge_index: np.ndarray,
        current_idx: int,
    ) -> None:
        self._h5_paths.append(self._sample_path(sample_name))
        self._sample_n_spots.append(n_spots)
        self._edge_indices.append(edge_index)
        self.sample_indices.append((current_idx, current_idx + n_spots, sample_name))
        self.sample_name_by_id.append(sample_name)

    def _init_lazy(self) -> None:
        current_idx = 0
        coords_norm_list, sample_id_list, local_idx_list = [], [], []
        for sample_idx, sample_name in enumerate(self.sample_names):
            h5_path = self._sample_path(sample_name)
            if not os.path.exists(h5_path):
                print(f'Warning: file not found: {h5_path}')
                continue
            with h5py.File(h5_path, 'r') as f:
                n_spots, feature_dim, num_genes, coords_norm, edge_index, backbone = self._read_sample_metadata(f)
                self._validate_sample_schema(
                    sample_name=sample_name,
                    feature_dim=feature_dim,
                    num_genes=num_genes,
                    feature_backbone=backbone,
                )
            self._register_sample(
                sample_idx=sample_idx,
                sample_name=sample_name,
                n_spots=n_spots,
                edge_index=edge_index,
                current_idx=current_idx,
            )
            coords_norm_list.append(coords_norm)
            sample_id_list.append(np.full((n_spots,), sample_idx, dtype=np.int64))
            local_idx_list.append(np.arange(n_spots, dtype=np.int64))
            current_idx += n_spots

        if not self.sample_indices:
            raise ValueError('No valid data files were found.')
        self._total_spots = current_idx
        self.coords_norm = np.concatenate(coords_norm_list, axis=0)
        self.sample_id = np.concatenate(sample_id_list, axis=0)
        self.local_idx = np.concatenate(local_idx_list, axis=0)
    def _init_preload(self) -> None:
        current_idx = 0
        local_list, macro_list, coords_list, coords_norm_list, genes_list = [], [], [], [], []
        sample_id_list, local_idx_list = [], []
        for sample_idx, sample_name in enumerate(self.sample_names):
            h5_path = self._sample_path(sample_name)
            if not os.path.exists(h5_path):
                print(f'Warning: file not found: {h5_path}')
                continue
            with h5py.File(h5_path, 'r') as f:
                expr_key = self._expr_key(f)
                n_spots, feature_dim, num_genes, coords_norm, edge_index, backbone = self._read_sample_metadata(f)
                local = np.asarray(f['local_features'][:], dtype=np.float32)
                macro = np.asarray(f['macro_features'][:], dtype=np.float32)
                coords = np.asarray(f['coords'][:], dtype=np.float32)
                genes = np.asarray(f[expr_key][:], dtype=np.float32)
                self._validate_sample_schema(
                    sample_name=sample_name,
                    feature_dim=feature_dim,
                    num_genes=num_genes,
                    feature_backbone=backbone,
                )

            self._register_sample(
                sample_idx=sample_idx,
                sample_name=sample_name,
                n_spots=n_spots,
                edge_index=edge_index,
                current_idx=current_idx,
            )
            local_list.append(local)
            macro_list.append(macro)
            coords_list.append(coords)
            coords_norm_list.append(coords_norm)
            genes_list.append(genes)
            sample_id_list.append(np.full((n_spots,), sample_idx, dtype=np.int64))
            local_idx_list.append(np.arange(n_spots, dtype=np.int64))
            current_idx += n_spots

        if not self.sample_indices:
            raise ValueError('No valid data files were found.')
        self._total_spots = current_idx
        self.local_features = np.concatenate(local_list, axis=0)
        self.macro_features = np.concatenate(macro_list, axis=0)
        self.coords = np.concatenate(coords_list, axis=0)
        self.coords_norm = np.concatenate(coords_norm_list, axis=0)
        self.genes = np.concatenate(genes_list, axis=0)
        self.sample_id = np.concatenate(sample_id_list, axis=0)
        self.local_idx = np.concatenate(local_idx_list, axis=0)

    def _get_h5_file(self, h5_path: str) -> h5py.File:
        with self._cache_lock:
            if h5_path not in self._h5_cache:
                self._h5_cache[h5_path] = h5py.File(h5_path, 'r', swmr=True)
            return self._h5_cache[h5_path]

    def _find_sample_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        for sample_idx, (start, end, _name) in enumerate(self.sample_indices):
            if start <= global_idx < end:
                return sample_idx, global_idx - start
        raise IndexError(global_idx)

    def _base_result(self, sample_idx: int, local_idx: int) -> dict:
        return {
            'sample_idx': int(sample_idx),
            'local_idx': int(local_idx),
            'h5_path': self._h5_paths[sample_idx],
            'sample_n_spots': int(self._sample_n_spots[sample_idx]),
            'edge_index_full': torch.as_tensor(self._edge_indices[sample_idx], dtype=torch.long),
        }

    def _getitem_lazy(self, idx: int) -> dict:
        sample_idx, local_idx = self._find_sample_and_local_idx(int(idx))
        f = self._get_h5_file(self._h5_paths[sample_idx])
        expr_key = self._expr_key(f)
        local_feat = torch.tensor(f['local_features'][local_idx], dtype=torch.float32)
        macro_feat = torch.tensor(f['macro_features'][local_idx], dtype=torch.float32)
        result = self._base_result(sample_idx, local_idx)
        result.update({
            'local_features': local_feat,
            'macro_features': macro_feat,
            'coords': torch.tensor(f['coords'][local_idx], dtype=torch.float32),
            'coords_norm': torch.tensor(self.coords_norm[idx], dtype=torch.float32),
            'genes': torch.tensor(f[expr_key][local_idx], dtype=torch.float32),
        })
        return result

    def _getitem_preload(self, idx: int) -> dict:
        sample_idx = int(self.sample_id[idx])
        local_idx = int(self.local_idx[idx])
        local_feat = torch.tensor(self.local_features[idx], dtype=torch.float32)
        macro_feat = torch.tensor(self.macro_features[idx], dtype=torch.float32)
        gene_expr = torch.tensor(self.genes[idx], dtype=torch.float32)
        if self.transform:
            local_feat = self.transform(local_feat)
            macro_feat = self.transform(macro_feat)
        if self.target_transform:
            gene_expr = self.target_transform(gene_expr)
        result = self._base_result(sample_idx, local_idx)
        result.update({
            'local_features': local_feat,
            'macro_features': macro_feat,
            'coords': torch.tensor(self.coords[idx], dtype=torch.float32),
            'coords_norm': torch.tensor(self.coords_norm[idx], dtype=torch.float32),
            'genes': gene_expr,
        })
        return result

    def __len__(self) -> int:
        return int(self._total_spots)

    def __getitem__(self, idx: int) -> dict:
        
        
        
        micrograph_id = None
        if isinstance(idx, (tuple, list)):
            if len(idx) != 2:
                raise ValueError(f'Packed dataset index must be (spot_index, micrograph_id), got {idx!r}')
            idx, micrograph_id = int(idx[0]), int(idx[1])
        sample = self._getitem_lazy(int(idx)) if self.lazy_load else self._getitem_preload(int(idx))
        if micrograph_id is not None:
            sample['_micrograph_id'] = micrograph_id
        return sample

    @property
    def num_genes(self) -> int:
        return int(self._num_genes)

    @property
    def feature_dim(self) -> int:
        return int(self._feature_dim)

    @property
    def feature_backbone(self) -> Optional[str]:
        return self._feature_backbone

    @property
    def num_modules(self) -> int:
        return int(self._num_modules)

    @property
    def gene_to_module(self) -> Optional[np.ndarray]:
        return None if self._gene_to_module is None else np.asarray(self._gene_to_module, dtype=np.int64)

    @property
    def module_sizes(self) -> Optional[np.ndarray]:
        return None if self._module_sizes is None else np.asarray(self._module_sizes, dtype=np.int64)


def _batch_subgraph(samples: List[dict]) -> torch.Tensor:
    group_ids = [int(s.get('_micrograph_id', 0)) for s in samples]
    ordered_groups = list(dict.fromkeys(group_ids))
    edges = []

    for group_id in ordered_groups:
        positions = [i for i, value in enumerate(group_ids) if value == group_id]
        group = [samples[i] for i in positions]
        sample_ids = {int(s['sample_idx']) for s in group}
        if len(sample_ids) != 1:
            raise ValueError('Each packed micrograph must contain spots from exactly one slide.')

        n_full = int(group[0]['sample_n_spots'])
        local_idx = torch.as_tensor([int(s['local_idx']) for s in group], dtype=torch.long)
        batch_positions = torch.as_tensor(positions, dtype=torch.long)
        mapping = torch.full((n_full,), -1, dtype=torch.long)
        mapping[local_idx] = batch_positions
        edge_full = group[0]['edge_index_full'].long()
        if edge_full.numel() == 0:
            continue
        src, dst = edge_full[0], edge_full[1]
        mask = (mapping[src] >= 0) & (mapping[dst] >= 0)
        edges.append(torch.stack([mapping[src[mask]], mapping[dst[mask]]], dim=0).long())

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat(edges, dim=1)


def make_basic_collate_fn() -> Callable[[List[dict]], dict]:
    def _collate(samples: List[dict]) -> dict:
        batch = {
            'local_features': torch.stack([s['local_features'] for s in samples], dim=0),
            'macro_features': torch.stack([s['macro_features'] for s in samples], dim=0),
            'coords': torch.stack([s['coords'] for s in samples], dim=0),
            'coords_norm': torch.stack([s['coords_norm'] for s in samples], dim=0),
            'genes': torch.stack([s['genes'] for s in samples], dim=0),
        }
        return batch
    return _collate


def make_spatial_collate_fn(k_neighbors: int) -> Callable[[List[dict]], dict]:
    del k_neighbors

    def _collate(samples: List[dict]) -> dict:
        batch = make_basic_collate_fn()(samples)
        batch['edge_index'] = _batch_subgraph(samples)
        return batch
    return _collate


class SpatialKNNBatchSampler(Sampler[List[int]]):
    """Groups nearby spots from the same slide into each mini-batch."""

    def __init__(self, dataset: SpatialTranscriptomicsDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        if NearestNeighbors is None:
            raise ImportError('scikit-learn is required for SpatialKNNBatchSampler')
        if batch_size < 2:
            raise ValueError('batch_size must be >= 2 for graph batching')
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._neighbors_by_sample: List[np.ndarray] = []
        self._ranges: List[Tuple[int, int]] = []
        for start, end, _name in dataset.sample_indices:
            n_i = end - start
            coords = dataset.coords_norm[start:end].astype(np.float32)
            nn = NearestNeighbors(n_neighbors=min(self.batch_size, n_i))
            nn.fit(coords)
            _dist, idx = nn.kneighbors(coords)
            self._neighbors_by_sample.append(idx.astype(np.int64))
            self._ranges.append((start, end))

    def __iter__(self) -> Iterator[List[int]]:
        for sample_idx, (start, end) in enumerate(self._ranges):
            n_i = end - start
            if n_i == 0:
                continue
            perm = np.random.permutation(n_i) if self.shuffle else np.arange(n_i)
            unused = np.ones(n_i, dtype=bool)
            neighbors = self._neighbors_by_sample[sample_idx]
            for seed in perm:
                if not unused[int(seed)]:
                    continue
                chosen = []
                for j in neighbors[int(seed)]:
                    j = int(j)
                    if unused[j]:
                        chosen.append(j)
                        unused[j] = False
                    if len(chosen) >= self.batch_size:
                        break
                if len(chosen) < self.batch_size:
                    remaining = perm[unused[perm]]
                    for j in remaining[: self.batch_size - len(chosen)]:
                        j = int(j)
                        chosen.append(j)
                        unused[j] = False
                if chosen and not (self.drop_last and len(chosen) < self.batch_size):
                    yield (np.asarray(chosen, dtype=np.int64) + start).tolist()

    def __len__(self) -> int:
        total = 0
        for start, end in self._ranges:
            n_i = end - start
            total += int(n_i // self.batch_size) if self.drop_last else int(np.ceil(n_i / self.batch_size))
        return total


class PackedSpatialKNNBatchSampler(Sampler[List[tuple[int, int]]]):
    """Pack disconnected spatial subgraphs into one batch."""

    def __init__(
        self,
        dataset: SpatialTranscriptomicsDataset,
        micro_batch_size: int,
        pack_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if int(pack_size) < 1:
            raise ValueError('pack_size must be >= 1')
        self.base_sampler = SpatialKNNBatchSampler(
            dataset,
            batch_size=int(micro_batch_size),
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
        )
        self.pack_size = int(pack_size)

    def __iter__(self) -> Iterator[List[tuple[int, int]]]:
        packed: List[tuple[int, int]] = []
        micrograph_id = 0
        for microbatch in self.base_sampler:
            packed.extend((int(idx), micrograph_id) for idx in microbatch)
            micrograph_id += 1
            if micrograph_id >= self.pack_size:
                yield packed
                packed = []
                micrograph_id = 0
        if packed:
            yield packed

    def __len__(self) -> int:
        return int(np.ceil(len(self.base_sampler) / self.pack_size))


def create_dataloaders(
    data_dir: str,
    train_samples: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    k_neighbors: int = 10,
    spatial_batching: bool = True,
    lazy_load: bool = False,
    gene_module_dir: Optional[str] = None,
    spatial_pack_size: int = 1,
) -> TorchDataLoader:
    train_dataset = SpatialTranscriptomicsDataset(data_dir, train_samples, lazy_load=lazy_load, gene_module_dir=gene_module_dir)
    collate_fn = make_spatial_collate_fn(k_neighbors=int(k_neighbors)) if spatial_batching else make_basic_collate_fn()
    persistent_workers = bool(num_workers and int(num_workers) > 0)
    prefetch_factor = 2 if persistent_workers else None
    spatial_pack_size = max(1, int(spatial_pack_size))

    common = dict(
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        common['prefetch_factor'] = prefetch_factor

    if spatial_batching:
        train_batch_sampler = (
            PackedSpatialKNNBatchSampler(
                train_dataset,
                micro_batch_size=batch_size,
                pack_size=spatial_pack_size,
                shuffle=True,
            )
            if spatial_pack_size > 1
            else SpatialKNNBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
        )
        train_loader = TorchDataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            **common,
        )
    else:
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            **common,
        )

    return train_loader
