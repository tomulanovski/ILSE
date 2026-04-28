# ILSE: Inter-Layer Structural Encoders

Code for *Improving LLM Predictions via Inter-Layer Structural Encoders*
(Ulanovski, Blyachman, Bechler-Speicher).

- Preprint: arXiv:XXXX.XXXXX
- Short version: GRaM Tiny Papers Workshop @ ICLR 2026

ILSE aggregates representations from all layers of a frozen LLM into a single downstream
representation by treating each layer's pooled hidden state as a node in a small graph and
learning a fusion encoder over those nodes.

## Methods

Three layer-aggregation encoders are implemented:

- **Cayley-Encoder** — GIN/GCN over a Cayley graph of `SL(2, Z_n)`, padded with virtual nodes when the graph is larger than the layer count.
- **FC-Encoder** — GIN/GCN over a fully-connected layer graph.
- **Set-Encoder** — DeepSet over the layer representations.

Baselines: Last-Layer, Best-Layer, MLP (last / best layer), Weighted (ELMo-style),
DWAtt with a 256-dim input projection, and LoRA fine-tuning.

## Installation

```bash
conda create -n ilse python=3.10 && conda activate ilse
pip install -r requirements.txt
```

A CUDA-capable GPU is required for embedding extraction and LoRA training.

## Usage

The standard pipeline is **precompute → train → evaluate**. Embeddings are extracted once
per LLM and reused across all training runs and Optuna trials.

### 1. Precompute layer-wise embeddings

```bash
# Classification
python -m experiments.utils.precompute.precompute_pipeline \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification EmotionClassification \
            MTOPDomainClassification MTOPIntentClassification PoemSentimentClassification \
    --output_dir ./precomputed_embeddings --pooling_method mean --batch_size 256

# STS
python -m experiments.utils.precompute.precompute_sts \
    --task STSBenchmark --model_family Pythia --model_size 410m \
    --output_dir ./precomputed_embeddings_sts --batch_size 256
```

Replace `--model_family Pythia --model_size 410m` with `Gemma2 2B` or `Llama3 8B` for the
other models reported in the paper. See
`experiments/utils/model_definitions/text_automodel_wrapper.py` for the full list of
supported HuggingFace LLM families.

### 2. Train

Classification:

```bash
python -m experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed \
    --task Banking77Classification \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder gin --graph_type cayley \
    --gin_hidden_dim 256 --gin_layers 2 --gin_mlp_layers 1 \
    --node_to_choose mean --dropout 0.1 \
    --epochs 50 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 \
    --save_dir ./saved_models --seed 42
```

STS:

```bash
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

To switch encoder, change `--encoder gin` to `mlp`, `weighted`, `deepset`, or `dwatt`,
and `--graph_type cayley` to `fully_connected`. `--gin_mlp_layers 0` makes the GIN a GCN.
When `--graph_type cayley`, the trainer auto-applies `pool_real_nodes_only=True` and
`train_eps=True`.

The corresponding live-LLM scripts (`basic_gin_trainer.py`, `sts_gin_trainer.py`) skip
the precompute step but run the LLM forward every epoch — slower per trial.

### 3. LoRA fine-tuning baseline

LoRA fine-tunes the base LLM directly and does not use precomputed embeddings:

```bash
# Classification
python -m experiments.utils.model_definitions.gnn.lora_trainer \
    --task Banking77Classification --model_family Gemma2 --model_size 2B \
    --lora_r 2 --lora_alpha 16 --lora_dropout 0.1 \
    --epochs 20 --batch_size 32 --lr 1e-3 --save_dir ./saved_models

# STS
python -m experiments.utils.model_definitions.gnn.sts_lora_trainer \
    --task STSBenchmark --model_family Gemma2 --model_size 2B \
    --lora_r 2 --lora_alpha 16 --lora_dropout 0.1 \
    --epochs 20 --batch_size 32 --lr 1e-3 --save_dir ./saved_models
```

LoRA outputs are a `.pt` checkpoint plus a sibling `_adapter/` directory; both are needed
at evaluation time.

### 4. Evaluate on MTEB

Classification:

```bash
python -m scripts_and_jobs.scripts.eval.mteb_evaluator \
    --model_path ./saved_models/<file>.pt \
    --model_family Pythia --model_size 410m \
    --tasks Banking77Classification \
    --output_dir ./mteb_results
```

STS (STSBenchmark + zero-shot transfer to STS12–16, BIOSSES, SICK-R):

```bash
python -m scripts_and_jobs.scripts.eval.evaluate_sts_model \
    --model_path ./saved_models/<sts_file>.pt \
    --model_family Pythia --model_size 410m \
    --encoder gin --config cayley \
    --tasks STSBenchmark STS12 STS13 STS14 STS15 STS16 BIOSSES SICK-R \
    --output_dir ./mteb_results/sts
```

The classification evaluator auto-detects the method from the filename prefix (`gin_`,
`mlp_`, `weighted_`, `deepset_`, `dwatt_`, `lora_`); override with `--model_type` if needed.

For raw last-layer / per-layer baselines without training:

```bash
python MTEB-Harness.py --model_family Pythia --model_size 410m \
    --evaluation_layer -1 --purpose run_tasks --filter_tasks Banking77Classification
```

## Hyperparameter search

Each method has an Optuna trial script under
`experiments/utils/model_definitions/gnn/optuna_runs/`. Trials share a study via a
PostgreSQL backend:

```bash
python -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_gin_precomputed \
    --study_name Banking77_Pythia410m_cayley \
    --task Banking77Classification \
    --model_family Pythia --model_size 410m \
    --embeddings_dir ./precomputed_embeddings/Pythia_410m_mean_pooling \
    --encoder gin --filter_graph_type cayley \
    --storage_url postgresql://user:pass@host:5432/optuna --n_trials 50
```

Equivalent scripts exist for STS (`run_optuna_trial_sts_gin_precomputed.py`) and LoRA
(`run_optuna_trial_lora.py`, `run_optuna_trial_sts_lora.py`).

## SLURM orchestration

`pipeline.py` automates the full workflow on a SLURM cluster:

```bash
python pipeline.py postgres            # one-time: launch Optuna's Postgres backend
python pipeline.py precompute --model Pythia-410m --submit
python pipeline.py optuna     --model Pythia-410m --methods gin mlp weighted --submit
python pipeline.py train      --model Pythia-410m --submit
python pipeline.py eval       --model Pythia-410m --submit
```

STS commands mirror these with the `sts-` prefix. Run `python pipeline.py --help` for all
subcommands and flags.

## Models and tasks

Models reported in the paper: Pythia-410M, Gemma2-2B, Llama3-8B. The scaling analysis
sweeps Pythia 14M through 2.8B.

Tasks: Banking77, Emotion, MTOPDomain, MTOPIntent, PoemSentiment (classification);
STSBenchmark + zero-shot transfer to STS12–16, BIOSSES, SICK-R (STS).

## Project layout

```
experiments/utils/
├── model_definitions/
│   ├── text_automodel_wrapper.py          # LLM loader / model registry
│   └── gnn/
│       ├── gnn_models.py                  # Encoder architectures
│       ├── gnn_datasets.py                # Datasets and graph construction
│       ├── basic_gin_trainer{,_precomputed}.py
│       ├── sts_gin_trainer{,_precomputed}.py
│       ├── lora_trainer.py / sts_lora_trainer.py
│       └── optuna_runs/                   # Per-method Optuna trial scripts
└── precompute/                            # Layer-wise embedding extraction
scripts_and_jobs/scripts/eval/             # MTEB evaluation wrappers
MTEB-Harness.py                            # Single-layer baselines
pipeline.py                                # SLURM workflow
```

## Citation

```bibtex
@misc{ulanovski2026ilse,
  title         = {Improving {LLM} Predictions via Inter-Layer Structural Encoders},
  author        = {Ulanovski, Tom and Blyachman, Eyal and Bechler-Speicher, Maya},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```

## Acknowledgments

`MTEB-Harness.py` was adapted from
[OFSkean/information_flow](https://github.com/OFSkean/information_flow). DWAtt is
re-implemented from ElNokrashy et al. (2024). LoRA uses
[Hugging Face PEFT](https://github.com/huggingface/peft).
