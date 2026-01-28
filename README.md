# ILSE: Inter Layer Structural Encoders

This repository contains the code for **ILSE**, a framework for structured layer-fusion in frozen Large Language Models. ILSE introduces three topological regimes for aggregating layer-wise embeddings:

- **Set-Encoder**: Permutation-invariant encoder with no inter-layer edges (empty graph)
- **FC-Encoder**: Dense graph connecting all layer embeddings (fully-connected)
- **Cayley-Encoder**: Sparse, 4-regular expander graph from SL(2, Z_n)

Additionally, we compare against:
- **MLP**: Multi-layer perceptron on last-layer embeddings
- **Weighted**: Learned scalar layer weighting (ELMo-style)
- **DWAtt**: Depth-Wise Attention baseline (ElNokrashy et al., 2024)
- **Single Layer**: Last-layer and best single-layer baselines

## Installation

### 1. Clone and setup environment

```bash
git clone <repository-url>
cd s_fusion
conda create -n s_fusion python=3.10
conda activate s_fusion
pip install -r requirements.txt
```

### 2. Verify GPU availability

ILSE requires a CUDA-capable GPU for embedding extraction and training.

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Reproducing Results

The pipeline has three stages: **(1) Precompute** layer-wise embeddings, **(2) Train** the fusion model, **(3) Evaluate** on MTEB test sets.

### Step 1: Precompute Layer-wise Embeddings (one-time)

Extract embeddings from all transformer layers of a frozen LLM and save to HDF5:

All scripts must be run from the project root directory using `python -m` module syntax:

```bash
# Classification tasks
python -m experiments.utils.precompute.precompute_pipeline \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification EmotionClassification MTOPDomainClassification \
           MTOPIntentClassification PoemSentimentClassification \
    --output_dir ./precomputed_embeddings \
    --pooling_method mean --batch_size 256

# STS tasks
python -m experiments.utils.precompute.precompute_sts \
    --task STSBenchmark --model_family Pythia --model_size 410m \
    --output_dir ./precomputed_embeddings_sts --batch_size 256
```

Supported models: `Pythia` (410m, 2.8b), `TinyLlama` (1.1B), `Llama3` (8B).

### Step 2: Train ILSE Models

Train using precomputed embeddings (no LLM needed at this stage):

```bash
# --- Cayley-Encoder (our best method) ---
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder gin --graph_type cayley \
    --gin_hidden_dim 256 --gin_layers 2 --gin_mlp_layers 1 \
    --node_to_choose mean \
    --dropout 0.1 --epochs 50 --batch_size 64 \
    --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# --- Cayley-Encoder (GCN mode) ---
# Same as above but with --gin_mlp_layers 0

# --- FC-Encoder (fully-connected graph) ---
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder gin --graph_type fully_connected \
    --gin_hidden_dim 256 --gin_layers 2 --gin_mlp_layers 1 \
    --node_to_choose mean \
    --dropout 0.1 --epochs 50 --batch_size 64 \
    --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# --- Set-Encoder (empty graph, permutation-invariant) ---
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder deepset --deepset_pooling_type mean \
    --deepset_pre_pooling_layers 1 --deepset_post_pooling_layers 1 \
    --dropout 0.1 --epochs 50 --batch_size 64 \
    --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# --- MLP Baseline ---
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder mlp --mlp_input last --mlp_layers 2 --mlp_hidden_dim 256 \
    --dropout 0.1 --epochs 50 --batch_size 64 \
    --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# --- Weighted Baseline (ELMo-style) ---
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder weighted \
    --epochs 50 --batch_size 64 \
    --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# --- DWAtt Baseline ---
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder dwatt --dwatt_hidden_dim 256 \
    --dropout 0.1 --epochs 50 --batch_size 64 \
    --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42
```

For **STS tasks**, use `sts_gin_trainer_precomputed.py` with the same encoder options (note: requires `--model_family` and `--model_size`):

```bash
# --- STS: Cayley-Encoder ---
python -m experiments.utils.model_definitions.gnn.sts_gin_trainer_precomputed \
    --task STSBenchmark \
    --embeddings_dir ./precomputed_embeddings_sts/Pythia_410m_mean_pooling \
    --model_family Pythia --model_size 410m \
    --encoder gin --graph_type cayley \
    --gin_hidden_dim 256 --gin_layers 2 --gin_mlp_layers 1 \
    --node_to_choose mean \
    --dropout 0.1 --epochs 50 --batch_size 64 \
    --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# Other encoders: same as classification, just swap the script and add --model_family/--model_size
```

### Step 3: Evaluate on MTEB Test Sets

Evaluation requires the frozen LLM to encode test texts at inference time.

**Classification tasks:**

```bash
python -m scripts_and_jobs.scripts.eval.mteb_evaluator \
    --model_path ./saved_models/<model_file>.pt \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification \
    --output_dir ./mteb_results
```

The evaluator auto-detects model type from the filename (e.g., `gin_`, `mlp_`, `weighted_`, `deepset_`, `dwatt_`). Override with `--model_type` if needed.

**STS tasks** (evaluates on STSBenchmark test + zero-shot transfer to STS12-16, BIOSSES, SICK-R):

```bash
python -m scripts_and_jobs.scripts.eval.evaluate_sts_model \
    --model_path ./saved_models/<sts_model_file>.pt \
    --model_family Pythia --model_size 410m \
    --encoder gin --config cayley \
    --tasks STSBenchmark STS12 STS13 STS14 STS15 STS16 BIOSSES SICK-R \
    --output_dir ./results/sts_results
```

### Single-Layer Baselines

Evaluate raw LLM layer embeddings (no training required):

```bash
# Last layer baseline
python MTEB-Harness.py \
    --model_family Pythia --model_size 410m \
    --evaluation_layer -1 --purpose run_tasks \
    --filter_tasks Banking77Classification

# Specific layer (e.g., layer 10)
python MTEB-Harness.py \
    --model_family Pythia --model_size 410m \
    --evaluation_layer 10 --purpose run_tasks \
    --filter_tasks Banking77Classification
```

## Hyperparameter Search (Optional)

For comprehensive hyperparameter optimization using Optuna (requires PostgreSQL):

```bash
python -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_gin_precomputed \
    --study_name Banking77_Pythia410m_cayley \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --task Banking77Classification \
    --model_family Pythia --model_size 410m \
    --encoder gin --filter_graph_type cayley \
    --storage_url postgresql://user:pass@host:5432/optuna \
    --n_trials 50
```

The default hyperparameters in the training scripts above correspond to the best configurations found via Optuna search as reported in the paper (Appendix F).

## Key Hyperparameters

| Parameter | Search Space | Description |
|-----------|-------------|-------------|
| `gin_layers` | {1, 2} | Number of message passing layers |
| `gin_mlp_layers` | {0, 1, 2} | MLP layers inside GIN (0 = GCN mode) |
| `gin_hidden_dim` | {256} | Hidden dimension (fixed) |
| `dropout` | {0.0, 0.1, 0.2, 0.3} | Dropout rate |
| `lr` | {1e-4, 1e-3} | Learning rate |
| `weight_decay` | {1e-4, 1e-3} | Weight decay |
| `node_to_choose` | {mean, sum} | Pooling method over layer nodes |
| `train_eps` | {True, False} | Learnable epsilon in GIN self-loop (auto-enabled for Cayley) |

## Supported Tasks

### Classification (5 tasks)
Banking77Classification, EmotionClassification, MTOPDomainClassification, MTOPIntentClassification, PoemSentimentClassification

### Semantic Textual Similarity (8 tasks)
STSBenchmark (training), STS12-STS16 (zero-shot eval), BIOSSES (zero-shot eval), SICK-R (zero-shot eval)

## Supported Models

| Model | Layers | Hidden Dim | Parameters |
|-------|--------|-----------|------------|
| Pythia-410m | 24 | 1024 | 410M |
| TinyLlama-1.1B | 23 | 2048 | 1.1B |
| Llama3-8B | 33 | 4096 | 8B |

Scaling analysis (Section 4.4) additionally uses Pythia-14m, 70m, 160m, 1b, 1.4b, and 2.8b.

## Project Structure

```
s_fusion/
├── experiments/utils/
│   ├── model_definitions/gnn/
│   │   ├── gnn_models.py                 # Model architectures (GIN, GCN, MLP, DeepSet, Weighted, DWAtt)
│   │   ├── gnn_datasets.py               # Dataset loading and graph construction
│   │   ├── basic_gin_trainer_precomputed.py  # Classification training
│   │   ├── sts_gin_trainer_precomputed.py    # STS training
│   │   └── optuna_runs/                  # Hyperparameter search scripts
│   ├── precompute/
│   │   ├── precompute_pipeline.py        # Extract classification embeddings
│   │   ├── precompute_sts.py             # Extract STS embeddings
│   │   └── h5_utils.py                   # HDF5 storage utilities
│   └── dataloaders/                      # Data loading utilities
├── scripts_and_jobs/scripts/eval/
│   ├── mteb_evaluator.py              # Classification evaluation (MTEB)
│   ├── evaluate_sts_model.py          # STS evaluation (MTEB)
│   └── *_wrapper.py                   # Per-method MTEB wrappers
├── pipeline.py                           # Pipeline orchestration (for SLURM clusters)
├── MTEB-Harness.py                       # Single-layer MTEB evaluation
└── requirements.txt
```

## SLURM Cluster Usage (Optional)

For large-scale experiments on SLURM clusters, see `pipeline.py` which automates job generation and submission:

```bash
cp .env.example .env  # Configure cluster-specific settings
python pipeline.py precompute --model Pythia-410m --submit
python pipeline.py optuna --model Pythia-410m --methods gin mlp weighted --submit
python pipeline.py train --model Pythia-410m --submit
python pipeline.py eval --model Pythia-410m --submit
```
