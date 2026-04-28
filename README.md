# ILSE: Inter-Layer Structural Encoders

> **Improving LLM Predictions via Inter-Layer Structural Encoders.**
> Tom Ulanovski, Eyal Blyachman, Maya Bechler-Speicher.
> ICLR 2026.

ILSE is a parameter-efficient post-training framework that aggregates representations from
**all layers** of a frozen LLM into a single, improved representation through structured
inter-layer interactions. ILSE treats each layer's pooled hidden state as a node in a small
graph and learns a fusion encoder over those nodes, recovering task-relevant signal that
single-layer or weighted-mean baselines miss — while adding **at most 0.1% extra parameters**
relative to the base LLM, and outperforming LoRA fine-tuning despite operating on frozen
representations.

The paper introduces three topological regimes:

- **Cayley-Encoder** *(recommended)* — sparse 4-regular expander graph over `SL(2, Z_n)`,
  with logarithmic diameter for bottleneck-free inter-layer communication.
- **FC-Encoder** — dense graph connecting every layer to every other layer.
- **Set-Encoder** — permutation-invariant DeepSet over layer representations (no edges).

Comparison baselines implemented in this repo:

- **Last-Layer** / **Best-Layer** — raw layer embeddings, no training.
- **MLP Last-Layer** / **MLP Best-Layer** — MLP probe over the final layer or the
  best-performing single layer (selected by `MTEB-Harness.py`).
- **Weighted** — ELMo-style learned scalar weighting of layers (Peters et al., 2018).
- **DWAtt** — Depth-Wise Attention (ElNokrashy et al., 2024), with a 256-dim input
  projection for fair parameter-count comparison.
- **LoRA** — PEFT fine-tuning of the base LLM (Hu et al., 2022).

---

## What this repository does

- **Extracts layer-wise embeddings** from any supported HuggingFace LLM and caches them to HDF5
  for fast reuse — train hundreds of fusion variants without re-running the LLM.
- **Trains seven fusion methods** (Cayley, FC, Set, MLP, Weighted, DWAtt, LoRA) on the cached
  embeddings, on classification and STS tasks.
- **Evaluates** trained models on the MTEB benchmark suite (5 classification tasks, 8 STS tasks
  including zero-shot transfer).
- **Hyperparameter search** via Optuna with a PostgreSQL backend for distributed sweeps.
- **SLURM cluster orchestration** for end-to-end paper-scale experiments (`pipeline.py`).
- **Two execution modes**: precomputed embeddings (recommended) or live LLM forward at training
  time (slower but no precompute step required).

---

## Installation

```bash
git clone <repository-url>
cd s_fusion
conda create -n ilse python=3.10
conda activate ilse
pip install -r requirements.txt
```

Verify GPU availability — ILSE requires a CUDA-capable GPU for embedding extraction and
LoRA fine-tuning:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

For Optuna sweeps with PostgreSQL backend (optional), install and start a local Postgres
server, then create a database `optuna`. The pipeline can launch one for you with
`python pipeline.py postgres`.

---

## Quickstart

A minimal end-to-end run on Pythia-410m, the **Banking77** classification task, with the
Cayley-Encoder:

```bash
# 1. Precompute layer-wise embeddings (one-time per model)
python -m experiments.utils.precompute.precompute_pipeline \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification \
    --output_dir ./precomputed_embeddings \
    --pooling_method mean --batch_size 256

# 2. Train the Cayley-Encoder
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder gin --graph_type cayley \
    --gin_hidden_dim 256 --gin_layers 2 --gin_mlp_layers 1 \
    --node_to_choose mean --dropout 0.1 \
    --epochs 50 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# 3. Evaluate on the MTEB test split
python -m scripts_and_jobs.scripts.eval.mteb_evaluator \
    --model_path ./saved_models/gin_Banking77Classification_Pythia_410m_cayley.pt \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification \
    --output_dir ./mteb_results
```

For STS tasks, replace the trainer with `sts_gin_trainer_precomputed.py` and the evaluator
with `evaluate_sts_model.py` — see [STS](#sts-tasks) below.

---

## Two execution modes

This repository supports two ways to train fusion models on top of a frozen LLM:

| Mode | Script suffix | When to use |
|---|---|---|
| **Precomputed** (recommended) | `*_precomputed.py` | Run the LLM once to dump all-layer embeddings to HDF5, then train any number of fusion variants without re-loading the LLM. Optuna sweeps are 10–100× faster this way. |
| **Live LLM** | `basic_gin_trainer.py`, `sts_gin_trainer.py` | The LLM forward runs every epoch. Simpler (no HDF5 step) but slow; use only for one-off runs where the precompute time would be wasted. |

**LoRA is a special case** — it fine-tunes the base LLM's parameters (via PEFT adapters), so
it cannot consume precomputed embeddings. `lora_trainer.py` and `sts_lora_trainer.py` always
run the LLM live.

The rest of this README assumes the precomputed path. To run without precomputing, swap the
script name (`basic_gin_trainer_precomputed.py` → `basic_gin_trainer.py`), drop
`--embeddings_dir`, and the remaining flags are largely identical (live trainers expose
`--pool_real_nodes_only` and `--train_eps` as explicit flags rather than auto-deriving them
from `--graph_type`).

---

## Step 1 — Precompute layer-wise embeddings

Embeddings are dumped per task to
`{output_dir}/{Model_Size}_{pooling}_pooling/{TaskName}.h5`.

```bash
# Classification — extract for all five paper tasks at once
python -m experiments.utils.precompute.precompute_pipeline \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification EmotionClassification \
            MTOPDomainClassification MTOPIntentClassification \
            PoemSentimentClassification \
    --output_dir ./precomputed_embeddings \
    --pooling_method mean --batch_size 256

# STS — separate script (handles sentence-pair structure)
python -m experiments.utils.precompute.precompute_sts \
    --task STSBenchmark --model_family Pythia --model_size 410m \
    --output_dir ./precomputed_embeddings_sts --batch_size 256
```

Swap `--model_family Pythia --model_size 410m` for any supported model — see
[Supported models](#supported-models). Common alternates:

```bash
# Gemma2 (uses eager attention internally to avoid NaN with padding)
... --model_family Gemma2 --model_size 2B ...

# Llama3
... --model_family Llama3 --model_size 8B ...
```

---

## Step 2 — Train a fusion model

All trainers below read from precomputed HDF5 embeddings. Outputs are saved as
`{method}_{task}_{model_family}_{model_size}_{config}.pt`.

### Classification

```bash
# Cayley-Encoder (our recommended method).
# When --graph_type cayley is set, the trainer auto-applies pool_real_nodes_only=True
# and train_eps=True to match the paper's setup.
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder gin --graph_type cayley \
    --gin_hidden_dim 256 --gin_layers 2 --gin_mlp_layers 1 \
    --node_to_choose mean --dropout 0.1 \
    --epochs 50 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42

# FC-Encoder — same script, --graph_type fully_connected.
# Set-Encoder — same script, --encoder deepset (no graph at all).
# MLP / Weighted / DWAtt — --encoder mlp / weighted / dwatt with method-specific flags.
```

GCN ablation is a single-flag flip: `--gin_mlp_layers 0` turns the GIN into a vanilla GCN.

### STS tasks

```bash
# Cayley-Encoder for STS — note the additional --model_family / --model_size flags
python -m experiments.utils.model_definitions.gnn.sts_gin_trainer_precomputed \
    --task STSBenchmark \
    --embeddings_dir ./precomputed_embeddings_sts/Pythia_410m_mean_pooling \
    --model_family Pythia --model_size 410m \
    --encoder gin --graph_type cayley \
    --gin_hidden_dim 256 --gin_layers 2 --gin_mlp_layers 1 \
    --node_to_choose mean --dropout 0.1 \
    --epochs 50 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42
```

### LoRA baseline (no precompute)

LoRA fine-tunes the base LLM directly via PEFT adapters; the layer-aggregation framework
is not involved. The classification and STS variants are separate scripts:

```bash
# Classification
python -m experiments.utils.model_definitions.gnn.lora_trainer \
    --task Banking77Classification \
    --model_family Gemma2 --model_size 2B \
    --lora_r 2 --lora_alpha 16 --lora_dropout 0.1 \
    --epochs 20 --batch_size 32 --lr 5e-4 \
    --save_dir ./saved_models --seed 42

# STS
python -m experiments.utils.model_definitions.gnn.sts_lora_trainer \
    --task STSBenchmark \
    --model_family Gemma2 --model_size 2B \
    --lora_r 2 --lora_alpha 16 --lora_dropout 0.1 \
    --epochs 20 --batch_size 32 --lr 5e-4 \
    --save_dir ./saved_models --seed 42
```

LoRA outputs are saved as a `.pt` checkpoint **plus a sibling `_adapter/` directory**
(PEFT adapter weights). Both are required at evaluation time.

---

## Step 3 — Evaluate on MTEB

### Classification

```bash
python -m scripts_and_jobs.scripts.eval.mteb_evaluator \
    --model_path ./saved_models/gin_Banking77Classification_Pythia_410m_cayley.pt \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification \
    --output_dir ./mteb_results
```

The evaluator auto-detects the method from the filename prefix (`gin_`, `mlp_`, `weighted_`,
`deepset_`, `dwatt_`, `lora_`). Override with `--model_type` if needed.

### STS

`evaluate_sts_model.py` runs the chosen STS encoder on **STSBenchmark** and zero-shot
transfers to STS12–STS16, BIOSSES, and SICK-R:

```bash
python -m scripts_and_jobs.scripts.eval.evaluate_sts_model \
    --model_path ./saved_models/gin_STSBenchmark_Pythia_410m_cayley_mean.pt \
    --model_family Pythia --model_size 410m \
    --encoder gin --config cayley \
    --tasks STSBenchmark STS12 STS13 STS14 STS15 STS16 BIOSSES SICK-R \
    --output_dir ./mteb_results/sts
```

### Single-layer baselines

`MTEB-Harness.py` evaluates raw layer embeddings without any training — useful for
last-layer and best-single-layer comparisons:

```bash
# Last layer
python MTEB-Harness.py \
    --model_family Pythia --model_size 410m \
    --evaluation_layer -1 --purpose run_tasks \
    --filter_tasks Banking77Classification

# Specific layer (e.g. layer 10)
python MTEB-Harness.py \
    --model_family Pythia --model_size 410m \
    --evaluation_layer 10 --purpose run_tasks \
    --filter_tasks Banking77Classification
```

---

## Hyperparameter search (Optuna)

Each method has its own Optuna trial script under `experiments/utils/model_definitions/gnn/optuna_runs/`.
A PostgreSQL backend allows multiple trial workers to share a single study:

```bash
# Classification GIN sweep
python -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_gin_precomputed \
    --study_name Banking77_Pythia410m_cayley \
    --task Banking77Classification \
    --model_family Pythia --model_size 410m \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder gin --filter_graph_type cayley \
    --storage_url postgresql://user:pass@host:5432/optuna \
    --n_trials 50

# STS GIN sweep
python -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_sts_gin_precomputed \
    --study_name STS_Pythia410m_cayley \
    --task STSBenchmark \
    --model_family Pythia --model_size 410m \
    --embeddings_dir ./precomputed_embeddings_sts/Pythia_410m_mean_pooling \
    --graph_type cayley \
    --storage_url postgresql://user:pass@host:5432/optuna \
    --n_trials 50

# LoRA sweep (classification)
python -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_lora \
    --study_name Banking77_Gemma2_lora \
    --task Banking77Classification \
    --model_family Gemma2 --model_size 2B \
    --storage_url postgresql://user:pass@host:5432/optuna \
    --n_trials 30
```

All paper-default hyperparameters in [Step 2](#step-2--train-a-fusion-model) correspond to
best configurations from these sweeps (Appendix F of the paper).

---

## SLURM cluster orchestration

`pipeline.py` automates job generation and submission for the entire workflow:

```bash
# One-time: launch local Postgres for Optuna
python pipeline.py postgres

# End-to-end on Pythia-410m
python pipeline.py precompute --model Pythia-410m --submit
python pipeline.py optuna     --model Pythia-410m --methods gin mlp weighted --submit
python pipeline.py train      --model Pythia-410m --submit
python pipeline.py eval       --model Pythia-410m --submit
python pipeline.py summarize  --filter-model Pythia_410m --output pythia410m_results.csv

# STS workflow (mirrors classification commands)
python pipeline.py sts-precompute --model Pythia-410m --submit
python pipeline.py sts-optuna     --model Pythia-410m --methods gin --submit
python pipeline.py sts-train      --model Pythia-410m --submit
python pipeline.py sts-eval       --model Pythia-410m --submit
```

`pipeline.py --help` lists all subcommands. Cluster-specific paths and account names live
in a local `.env` file — see `pipeline.py` source for the variables it reads.

---

## Hyperparameter reference

The paper's main result configurations:

| Parameter | Search Space | Description |
|---|---|---|
| `gin_layers` | {1, 2} | Number of GIN message-passing layers |
| `gin_mlp_layers` | {0, 1, 2} | MLP layers inside GIN (0 = GCN) |
| `gin_hidden_dim` | {256} | Hidden dimension (fixed) |
| `dropout` | {0.0, 0.1, 0.2, 0.3} | Dropout rate |
| `lr` | {1e-4, 1e-3} | Learning rate |
| `weight_decay` | {1e-4, 1e-3} | Weight decay |
| `node_to_choose` | {mean, sum} | Pooling over layer-graph nodes |
| `train_eps` | {True, False} | Learnable ε in GIN aggregation (auto-True for Cayley) |
| `pool_real_nodes_only` | {True, False} | Skip Cayley virtual nodes during pooling (auto-True for Cayley) |

**Cayley-specific defaults**: when `--graph_type cayley` is set, the trainer/Optuna scripts
automatically set `pool_real_nodes_only=True` and `train_eps=True`. The Cayley graph is
built as the smallest `SL(2, Z_n)` that fits the LLM's layer count, with any remaining
slots padded as virtual nodes.

---

## Supported models

The paper reports results on:

| Family | Sizes | HuggingFace ID |
|---|---|---|
| **Pythia** | 410m (main), plus 14m / 70m / 160m / 1b / 1.4b / 2.8b for scaling analysis | `EleutherAI/pythia-{size}` |
| **Gemma2** | 2B | `google/gemma-2-2b` |
| **Llama3** | 8B | `meta-llama/Meta-Llama-3-8B` |

The wrapper (`experiments/utils/model_definitions/text_automodel_wrapper.py`) also accepts
several other HuggingFace LLM families (Mamba, Cerebras-GPT, BERT/RoBERTa, LLM2Vec, …) —
see `model_types` in that file if you want to extend experiments beyond the paper.

---

## Supported tasks

**Classification (5 MTEB tasks):** Banking77Classification, EmotionClassification,
MTOPDomainClassification, MTOPIntentClassification, PoemSentimentClassification.

**Semantic Textual Similarity (8 MTEB tasks):** STSBenchmark (training), STS12–STS16
(zero-shot), BIOSSES (zero-shot), SICK-R (zero-shot).

---

## Project structure

```
s_fusion/
├── experiments/utils/
│   ├── model_definitions/
│   │   ├── text_automodel_wrapper.py          # LLM loader / model registry
│   │   └── gnn/
│   │       ├── gnn_models.py                  # GIN/GCN, MLP, Weighted, DeepSet, DWAtt
│   │       ├── gnn_datasets.py                # Dataset loading + graph construction
│   │       ├── basic_gin_trainer.py           # Classification (live LLM)
│   │       ├── basic_gin_trainer_precomputed.py   # Classification (precomputed)
│   │       ├── sts_gin_trainer.py             # STS (live LLM)
│   │       ├── sts_gin_trainer_precomputed.py # STS (precomputed)
│   │       ├── lora_trainer.py                # Classification LoRA
│   │       ├── sts_lora_trainer.py            # STS LoRA
│   │       └── optuna_runs/                   # Per-method Optuna trial scripts
│   └── precompute/
│       ├── precompute_pipeline.py             # Classification embedding extraction
│       ├── precompute_sts.py                  # STS embedding extraction
│       └── h5_utils.py                        # HDF5 I/O
├── scripts_and_jobs/scripts/eval/
│   ├── mteb_evaluator.py                      # Classification MTEB eval
│   ├── evaluate_sts_model.py                  # STS MTEB eval
│   ├── gnn_wrapper.py / mlp_wrapper.py / ...  # Per-method MTEB-compatible wrappers
│   └── lora_wrapper.py                        # LoRA MTEB wrapper (loads base + adapter)
├── MTEB-Harness.py                            # Single-layer baselines
├── pipeline.py                                # SLURM workflow orchestration
└── requirements.txt
```

---

## Citation

If you use ILSE in your research, please cite:

```bibtex
@inproceedings{ulanovski2026ilse,
  title     = {Improving {LLM} Predictions via Inter-Layer Structural Encoders},
  author    = {Ulanovski, Tom and Blyachman, Eyal and Bechler-Speicher, Maya},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---

## Acknowledgments

The single-layer baseline harness (`MTEB-Harness.py`) was adapted from
[OFSkean/information_flow](https://github.com/OFSkean/information_flow). DWAtt is
re-implemented from ElNokrashy et al. (2024). LoRA fine-tuning uses
[Hugging Face PEFT](https://github.com/huggingface/peft).

This study was supported in part by a fellowship from the Edmond J. Safra Center for
Bioinformatics at Tel-Aviv University.
