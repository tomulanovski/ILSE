#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-general-pool
#SBATCH --account=your-account
#SBATCH --gres=gpu:A100:1
#SBATCH --output=job_logs/gin_eval_MTOPnew_Alltasks.out
#SBATCH --error=job_logs/gin_eval_MTOPnew_Alltasks.err

# Load environment variables from .env file (if available)
# Note: GNN_REPO_DIR should be set before running this script, or use absolute path in .env
if [ -f ".env" ]; then
    export $(cat ".env" | grep -v '^#' | xargs)
fi

# Use environment variables with fallbacks
CONDA_PATH=${GNN_CONDA_PATH:-/path/to/miniconda3}
REPO_DIR=${GNN_REPO_DIR:-$(pwd)}

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"


# Activate your specific environment
conda activate final_project 

# Verify the environment was activated
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to the repo directory
cd "${REPO_DIR}"


python -m scripts_and_jobs.scripts.eval.mteb_evaluator --model_family Pythia --model_size 410m --model_path "saved_models/MTOPIntentClassification_Pythia_410m_gin.pt" --tasks EmotionClassification AmazonCounterfactualClassification AskUbuntuDupQuestions BIOSSES Banking77Classification MTOPDomainClassification MTOPIntentClassification MassiveIntentClassification MassiveScenarioClassification MedrxivClusteringS2S SICK-R STS12 STS13 STS14 STS15 STS16 STS17 STSBenchmark SprintDuplicateQuestions TweetSentimentExtractionClassification --output_dir ./mteb_results_gnn_MTOP_best