# TRACE

TRACE predicts spatial gene expression from H&E images using frozen pathology
image features, spatial topology, gene-module composition, and count-scale
factorization.

The runnable workflow is:

```text
filtered h5ad files + whole-slide images
  -> local and macro image features + spatial graph
  -> gene modules fitted on training slides only
  -> TRACE training
  -> inference and evaluation on the held-out slide
```

## Repository layout

```text
gene_lists.json              Ordered target-gene lists
scripts/
  build_processed_data.py    Build one processed H5 file per slide
  build_gene_modules.py      Fit gene modules on training slides
  train_genar.py             Train TRACE
  infer_genar.py             Run inference and compute evaluation metrics
src/
  configs/example.yaml       Runnable example configuration
  data.py                    Dataset loading and spatial batching
  features.py                ResNet50, UNI, and UNI2-h feature extraction
  models/
    fdt.py                   Frequency-Decoupled Topology-aware Trunk
    gmcp.py                  MPN guide and GMCP decoder
    trace.py                 TRACE assembly and scale prediction
  training.py                Training objective and trainer
  utils.py                   Path and logging utilities
```

`gene_lists.json` contains ordered 200-gene panels for `her2st`, `prad`,
`kidney`, and `healthy_mouse_brain`. All slides in one experiment must use the
same genes in the same order.

## Method overview

### Inputs

For each spot, TRACE uses:

- `local_features`: a pooled feature from the local image crop;
- `macro_features`: a pooled feature from the larger contextual crop;
- `coords_norm`: slide-normalized `(x, y)` coordinates;
- `edge_index`: a symmetric spatial k-nearest-neighbor graph in COO format.

### FDT

FDT aggregates local and macro image features over the spatial graph, adds
Fourier coordinate features, and fuses the two semantic paths. A separate
high-frequency residual path performs graph band-pass processing and adaptive
band mixing. The semantic and high-frequency representations form `z_final`.

The implementation supports `pos_mlp` and `local_bandpass` residual paths. The
example configuration selects `local_bandpass`.

### GMCP

MPN predicts a prior over gene modules from the semantic representation. GMCP
combines this prior with `z_final` to estimate module composition and
within-module gene composition. Their product is a normalized gene
composition.

The gene-to-module assignment is fixed before model training and is fitted
from training-slide expression only.

### Count scale

A scale head predicts the library size, and TRACE reconstructs raw expression
as:

```text
pred_raw = gene_composition * predicted_library_size
```

### Training objective

The implemented objective contains:

- Smooth L1 loss on `log1p` raw expression;
- KL divergence for gene composition;
- weighted KL divergence for module composition;
- weighted Smooth L1 loss for `log1p` library size;
- optional log-MSE, gene-correlation, graph-frequency, and local-gradient
  terms.

The optional terms have zero weight in `example.yaml`.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The preprocessing pipeline supports three frozen image encoders:

| `feature_backbone` | Feature dimension | Weights |
| --- | ---: | --- |
| `resnet50` | 2048 | torchvision ImageNet-1K V2 |
| `uni` | 1024 | MahmoodLab/UNI |
| `uni2-h` | 1536 | MahmoodLab/UNI2-h |

UNI and UNI2-h require access to their official Hugging Face repositories.

```bash
huggingface-cli login
```

## 1. Prepare slide-level H5 files

Each input h5ad file must contain raw count-scale expression in `X` and the
final target genes in the published order. Spot coordinates must be present as
`pxl_col_in_fullres` and `pxl_row_in_fullres`, or in `obsm['spatial']`. Each
whole-slide image must have the same sample stem as its h5ad file and use the
`.tif` extension.

```bash
python scripts/build_processed_data.py \
  --h5ad_dir path/to/filtered_h5ad \
  --wsi_dir path/to/wsi \
  --out_dir processed_data_example \
  --dataset_name DATASET \
  --backbone uni2-h \
  --local_size 224 \
  --macro_size 1024 \
  --graph_k 10 \
  --batch_size 8
```

Use `--backbone resnet50`, `--backbone uni`, or `--backbone uni2-h`. Set the
same encoder in the configuration:

```yaml
data:
  feature_backbone: uni2-h
```

TRACE reads the native feature dimension from each processed H5 file. Training
and inference reject mixed backbones and a mismatch with
`data.feature_backbone`.

Each processed H5 file contains:

```text
barcodes
coords
coords_norm
edge_index
genes_count
gene_names
local_features
macro_features
```

Its attributes record the backbone, feature dimension, crop sizes, graph
settings, coordinate convention, and gene-order hash.

## 2. Define the train/test split

Edit `src/configs/example.yaml`:

```yaml
data:
  data_dir: processed_data_example
  train_samples: AUTO_EXCEPT_VAL_TEST
  test_samples:
    - TEST_SAMPLE
  gene_module_dir: processed_data_example/gene_modules_k10
```

With `AUTO_EXCEPT_VAL_TEST`, every `.h5` file in `data_dir` except the names in
`test_samples` is used for training. Replace `TEST_SAMPLE` with the held-out
sample stem. The same split is used by module construction and model training.

## 3. Build training-only gene modules

Gene modules must be fitted without the held-out test slide. Run:

```bash
python scripts/build_gene_modules.py \
  --config src/configs/example.yaml \
  --seed 42
```

The script performs the following operations:

1. resolves the training samples from the configuration;
2. excludes every configured test sample when automatic splitting is used;
3. concatenates `genes_count` from the training slides only;
4. applies per-spot library-size normalization to 10,000 followed by `log1p`;
5. transposes the matrix to obtain one expression profile per gene;
6. fits K-means with the configured `model.decoder.num_modules`.

The configured `gene_module_dir` then contains:

| File | Contents |
| --- | --- |
| `gene_to_module.npy` | Integer module ID for each gene in the published gene order |
| `module_sizes.npy` | Number of genes assigned to each module |
| `metadata.json` | Module count, gene count, seed, normalization, and training-sample names |

`gene_to_module.npy` and `module_sizes.npy` are required by both training and
inference. They are generated by this command before either stage is run.

## 4. Train TRACE

Review the example settings and run:

```bash
python scripts/train_genar.py \
  --config src/configs/example.yaml
```

When `training.checkpoint.save_last` is enabled, `last_model.pth` is overwritten
after each epoch in the configured checkpoint directory. Resume explicitly if
needed:

```bash
python scripts/train_genar.py \
  --config src/configs/example.yaml \
  --resume path/to/last_model.pth
```

## 5. Inference and evaluation

`auto:last` loads `last_model.pth` from the checkpoint directory specified in
the configuration.

```bash
python scripts/infer_genar.py \
  --config src/configs/example.yaml \
  --model_path auto:last \
  --test_sample TEST_SAMPLE \
  --output_file results/example_prediction.h5 \
  --save-target-for-eval
```

With `--save-target-for-eval`, the output includes the ground truth and the
following metrics computed on `log1p` expression:

- gene-wise Pearson correlation coefficient;
- spot-wise Pearson correlation coefficient;
- mean squared error;
- mean absolute error.

Without this flag, the prediction file contains predictions and metadata but
does not contain ground-truth expression or evaluation metrics.

`example.yaml` is a runnable example configuration. Dataset paths, the held-out
sample, and experiment settings must be set for the intended experiment.
