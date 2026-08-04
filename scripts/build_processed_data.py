"""Build processed TRACE H5 files from h5ad data and whole-slide images."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import h5py
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import UNIEncoder
from utils import project_root, resolve_path


def _parse_h5_compression(name: str | None) -> tuple[Optional[str], Optional[int]]:
    if name is None:
        return None, None
    n = str(name).lower().strip()
    if n in ('none', 'off', 'false', '0', ''):
        return None, None
    if n in ('gzip', 'lzf'):
        return n, None
    raise ValueError(f'Unsupported HDF5 compression: {name}. Use one of: none, lzf, gzip')


def _gene_order_hash(gene_names: List[str]) -> str:
    payload = '\n'.join(str(g) for g in gene_names).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _load_adata(h5ad_path: str):
    return ad.read_h5ad(h5ad_path, backed='r')


def _get_coords_xy(adata) -> np.ndarray:
    if 'pxl_col_in_fullres' in adata.obs.columns and 'pxl_row_in_fullres' in adata.obs.columns:
        x = np.asarray(adata.obs['pxl_col_in_fullres']).astype(np.float32)
        y = np.asarray(adata.obs['pxl_row_in_fullres']).astype(np.float32)
        return np.stack([x, y], axis=1)
    if 'spatial' in adata.obsm_keys():
        return np.asarray(adata.obsm['spatial']).astype(np.float32)
    raise KeyError("Could not find fullres coordinates in obs pxl_* or obsm['spatial']")


def _get_barcodes(adata) -> np.ndarray:
    return np.asarray([str(x).encode('utf-8') for x in list(adata.obs_names)], dtype='S')


def _materialize_genes_dense_float32(adata) -> np.ndarray:
    x = adata.X
    try:
        x = x.to_memory()
    except Exception:
        pass
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            x = x.toarray()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def _coords_norm(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    cmin = coords.min(axis=0, keepdims=True)
    cmax = coords.max(axis=0, keepdims=True)
    return ((coords - cmin) / (cmax - cmin + 1e-8)).astype(np.float32)


def _build_knn_graph(coords: np.ndarray, k: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    n = int(coords.shape[0])
    if n < 2:
        return np.empty((2, 0), dtype=np.int64)
    k_eff = int(min(max(int(k), 1), n - 1))
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean')
        nn.fit(coords)
        _dist, idx = nn.kneighbors(coords)
        idx = idx[:, 1:]
    except Exception:
        d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(d, np.inf)
        idx = np.argsort(d, axis=1)[:, :k_eff]

    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = idx.reshape(-1).astype(np.int64)
    
    edges = np.concatenate([np.stack([src, dst], axis=1), np.stack([dst, src], axis=1)], axis=0)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    if edges.shape[0] > 0:
        keep = np.ones(edges.shape[0], dtype=bool)
        keep[1:] = np.any(edges[1:] != edges[:-1], axis=1)
        edges = edges[keep]
    return edges.T.astype(np.int64)


def _crop_with_padding(img: Image.Image, center_xy: Tuple[float, float], size: int) -> Image.Image:
    cx, cy = center_xy
    half = int(size) // 2
    left = int(round(float(cx) - half))
    top = int(round(float(cy) - half))
    right = left + int(size)
    bottom = top + int(size)
    canvas = Image.new('RGB', (int(size), int(size)), (255, 255, 255))
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(img.width, right)
    src_bottom = min(img.height, bottom)
    if src_right <= src_left or src_bottom <= src_top:
        return canvas
    crop = img.crop((src_left, src_top, src_right, src_bottom)).convert('RGB')
    canvas.paste(crop, (src_left - left, src_top - top))
    return canvas


@torch.no_grad()
def _encode_batch(
    encoder: UNIEncoder,
    local_pils: List[Image.Image],
    macro_pils: List[Image.Image],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    local_t = torch.stack([encoder.transform(p) for p in local_pils], dim=0).to(device, non_blocking=True)
    macro_t = torch.stack([encoder.transform(p) for p in macro_pils], dim=0).to(device, non_blocking=True)
    if use_amp and device.type == 'cuda':
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            lf = encoder(local_t)
            mf = encoder(macro_t)
    else:
        lf = encoder(local_t)
        mf = encoder(macro_t)
    return lf.detach().float().cpu().numpy(), mf.detach().float().cpu().numpy()


def build_one_sample(
    sample: str,
    h5ad_dir: str,
    wsi_dir: str,
    out_dir: str,
    encoder: UNIEncoder,
    batch_size: int,
    local_size: int,
    macro_size: int,
    graph_k: int,
    dataset_name: str,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    crop_workers: int,
    compression: str = 'lzf',
    gzip_level: int = 4,
):
    h5ad_path = os.path.join(h5ad_dir, f'{sample}.h5ad')
    wsi_path = os.path.join(wsi_dir, f'{sample}.tif')
    out_path = os.path.join(out_dir, f'{sample}.h5')
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(h5ad_path)
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(wsi_path)

    print(f'\n== {sample} ==')
    print(f'h5ad: {h5ad_path}')
    print(f'wsi : {wsi_path}')
    print(f'out : {out_path}')

    adata = _load_adata(h5ad_path)
    coords = _get_coords_xy(adata).astype(np.float32)
    coords_n = _coords_norm(coords)
    barcodes = _get_barcodes(adata)
    genes = _materialize_genes_dense_float32(adata)
    gene_names_list = [str(x) for x in list(adata.var_names)]
    gene_names = np.asarray([g.encode('utf-8') for g in gene_names_list], dtype='S')
    gene_hash = _gene_order_hash(gene_names_list)
    edge_index = _build_knn_graph(coords, k=int(graph_k))

    n_spots = int(coords.shape[0])
    print(f'spots={n_spots}, genes={genes.shape[1]}, graph_edges={edge_index.shape[1]}, gene_order_hash={gene_hash[:12]}')

    img = Image.open(wsi_path)
    try:
        img.load()
    except Exception:
        pass
    print(f'wsi size={img.width}x{img.height}, mode={img.mode}')

    os.makedirs(out_dir, exist_ok=True)
    comp, comp_opts = _parse_h5_compression(compression)
    if comp == 'gzip':
        comp_opts = int(gzip_level)
    with h5py.File(out_path, 'w') as f_out:
        f_out.create_dataset('barcodes', data=barcodes)
        f_out.create_dataset('coords', data=coords, compression=comp, compression_opts=comp_opts)
        f_out.create_dataset('coords_norm', data=coords_n, compression=comp, compression_opts=comp_opts)
        f_out.create_dataset('edge_index', data=edge_index, compression=comp, compression_opts=comp_opts)
        f_out.create_dataset('genes_count', data=genes, compression=comp, compression_opts=comp_opts)
        f_out.create_dataset('gene_names', data=gene_names)

        ds_local = f_out.create_dataset(
            'local_features',
            shape=(n_spots, int(encoder.output_dim)),
            dtype=np.float32,
            compression=comp,
            compression_opts=comp_opts,
            chunks=(min(int(batch_size), max(n_spots, 1)), int(encoder.output_dim)),
        )
        ds_macro = f_out.create_dataset(
            'macro_features',
            shape=(n_spots, int(encoder.output_dim)),
            dtype=np.float32,
            compression=comp,
            compression_opts=comp_opts,
            chunks=(min(int(batch_size), max(n_spots, 1)), int(encoder.output_dim)),
        )

        f_out.attrs['sample_name'] = str(sample)
        f_out.attrs['dataset_name'] = str(dataset_name)
        f_out.attrs['feature_backbone'] = str(encoder.backbone_name)
        f_out.attrs['feature_dim'] = int(encoder.output_dim)
        f_out.attrs['feature_view'] = 'local_macro'
        f_out.attrs['graph_k'] = int(graph_k)
        f_out.attrs['graph_symmetric'] = True
        f_out.attrs['graph_self_loop'] = False
        f_out.attrs['coordinate_system'] = 'WSI full-resolution pixel coordinates, x=column y=row'
        f_out.attrs['local_patch_size'] = int(local_size)
        f_out.attrs['macro_patch_size'] = int(macro_size)
        f_out.attrs['gene_order_hash'] = gene_hash
        f_out.attrs['expression_field'] = 'genes_count'
        f_out.attrs['normalization_method'] = 'none; count-scale expression from filtered h5ad.X'

        crop_executor = ThreadPoolExecutor(max_workers=int(crop_workers)) if int(crop_workers) > 1 else None
        for start in range(0, n_spots, int(batch_size)):
            end = min(start + int(batch_size), n_spots)
            batch_xy = coords[start:end]

            def _make_views(xy):
                x, y = xy
                local = _crop_with_padding(img, (float(x), float(y)), int(local_size))
                if int(local_size) != 224:
                    local = local.resize((224, 224), resample=Image.BICUBIC)
                macro = _crop_with_padding(img, (float(x), float(y)), int(macro_size))
                macro = macro.resize((224, 224), resample=Image.BICUBIC)
                return local, macro

            if crop_executor is not None:
                views = list(crop_executor.map(_make_views, batch_xy))
                local_pils = [v[0] for v in views]
                macro_pils = [v[1] for v in views]
            else:
                views = [_make_views(xy) for xy in batch_xy]
                local_pils = [v[0] for v in views]
                macro_pils = [v[1] for v in views]

            lf, mf = _encode_batch(encoder, local_pils, macro_pils, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
            ds_local[start:end] = lf
            ds_macro[start:end] = mf
            if (start // int(batch_size)) % 10 == 0 or end == n_spots:
                print(f'  encoded {end}/{n_spots}')

        if crop_executor is not None:
            crop_executor.shutdown(wait=True)
        f_out.flush()

    try:
        adata.file.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description='Build TRACE processed H5 from filtered h5ad + WSI')
    ap.add_argument('--h5ad_dir', default='hest1k_datasets/her2st/st_filtered')
    ap.add_argument('--wsi_dir', default='hest1k_datasets/her2st/wsis')
    ap.add_argument('--out_dir', default='processed_data_her2st')
    ap.add_argument('--dataset_name', default='HER2ST')
    ap.add_argument('--samples', nargs='*', default=None)
    ap.add_argument('--backbone', default='uni2-h', choices=['resnet50', 'uni', 'uni2-h'])
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--local_size', type=int, default=224, help='Local crop size in full-resolution pixels before UNI transform.')
    ap.add_argument('--macro_size', type=int, default=1024)
    ap.add_argument('--graph_k', type=int, default=10)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16'])
    ap.add_argument('--crop_workers', type=int, default=8)
    ap.add_argument('--compression', default='lzf', choices=['none', 'lzf', 'gzip'])
    ap.add_argument('--gzip_level', type=int, default=4)
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    repo = project_root()
    h5ad_dir = resolve_path(args.h5ad_dir, base_dir=str(repo))
    wsi_dir = resolve_path(args.wsi_dir, base_dir=str(repo))
    out_dir = resolve_path(args.out_dir, base_dir=str(repo))

    if args.samples is None or len(args.samples) == 0:
        args.samples = sorted([p.stem for p in Path(h5ad_dir).glob('*.h5ad')])
        if not args.samples:
            raise ValueError(f'No .h5ad files found in h5ad_dir: {h5ad_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    use_amp = not bool(args.no_amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16

    encoder = UNIEncoder(backbone=args.backbone, pretrained=True)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    if args.backbone in ('uni', 'uni2-h') and not getattr(encoder, 'loaded_pretrained', False):
        raise RuntimeError(
            f'Backbone={args.backbone} requested pretrained weights, but loading failed. '
            'Check Hugging Face access and local weight paths.'
        )

    os.makedirs(out_dir, exist_ok=True)
    for sample in args.samples:
        out_path = os.path.join(out_dir, f'{sample}.h5')
        if args.overwrite and os.path.exists(out_path):
            os.remove(out_path)
        elif args.skip_existing and os.path.exists(out_path):
            print(f'\n== {sample} ==\n  skip: exists -> {out_path}')
            continue
        build_one_sample(
            sample=sample,
            h5ad_dir=h5ad_dir,
            wsi_dir=wsi_dir,
            out_dir=out_dir,
            encoder=encoder,
            batch_size=int(args.batch_size),
            local_size=int(args.local_size),
            macro_size=int(args.macro_size),
            graph_k=int(args.graph_k),
            dataset_name=str(args.dataset_name),
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            crop_workers=int(args.crop_workers),
            compression=str(args.compression),
            gzip_level=int(args.gzip_level),
        )
    print('\nDone. Next step: build GMCP gene modules from training slides only.')


if __name__ == '__main__':
    main()
