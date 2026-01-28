"""
MTEB evaluation harness for single-layer baselines.

This code was adapted from:
https://github.com/OFSkean/information_flow
"""
import argparse
import os
from itertools import product
import json
from pathlib import Path
import mteb


from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from experiments.utils.model_definitions.simple_aggregation_wrapper import SimpleAggregationWrapper

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils import logging
logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', type=str, default='mamba')
    parser.add_argument('--model_size', type=str, default='370m')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--evaluation_layer', type=int, default=-1, help='Layer to use for evaluation. -1 for the final layer. This is 0-indexed.')
    parser.add_argument('--base_results_path', type=str, default='experiments/results')
    parser.add_argument('--purpose', type=str, default='run_entropy_metrics', choices=['run_tasks', 'run_entropy_metrics', 'run_wikitext_metrics', 'download_datasets'])
    parser.add_argument('--raise_error', type=bool, default=False)
    parser.add_argument('--use_simple_aggregation', action='store_true', help='Use SimpleAggregation wrapper instead of standard model')
    parser.add_argument('--aggregation_method', type=str, default='mean', choices=['add', 'mean', 'max'], help='Aggregation method to use')
    parser.add_argument('--token_pooling_method', type=str, default='mean', choices=["mean", "first_hidden_state", "last_hidden_state"], help='Pooling method on tokens')
    parser.add_argument('--nodes_pooling_method', type=str, default='last', choices=['last', 'first', 'mean'], help='Pooling method on nodes')
    parser.add_argument('--num_gnn_layers', type=int, default=1, help='Number of GNN layers')
    parser.add_argument('--cayley_jumps', type=str, default='1,3', help='Cayley jumps')
    parser.add_argument('--graph_type', type=str, default='fully_connected', choices=['linear', 'fully_connected', 'virtual_node', 'cayley', 'cayley'], help='Graph construction type')
    parser.add_argument('--filter_tasks', type=str, default='all', help='Task filtering: "all", "classification_subset", or comma-separated task names')
    return parser.parse_args()


def list_done_tasks(results_root: Path, split: str = "test"):
    done = set()
    for jf in Path(results_root).glob("*.json"):
        try:
            with jf.open("r") as f:
                data = json.load(f)
            name = jf.stem
            if data.get("scores", {}).get(split):
                done.add(name)
        except Exception:
            pass
    return done


def main():
    args = parse_args()
    model_family = args.model_family
    model_size = args.model_size
    revision = args.revision
    evaluation_layer = args.evaluation_layer

    print(f"Running evaluation for {model_family} {model_size} {revision} layer {evaluation_layer}")
    model_specs = TextModelSpecifications(model_family, model_size, revision=revision)

    # handle tasks
    mteb_eng = mteb.get_benchmark("MTEB(eng)")

    # Apply task filtering based on --filter_tasks argument
    if args.filter_tasks == 'all':
        # Original filtering logic: exclude p2p, retrieval, and OOM tasks
        reduced_mteb_eng_tasks = [task for task in mteb_eng if task.metadata.category != 'p2p']
        reduced_mteb_eng_tasks = [task for task in reduced_mteb_eng_tasks if task.metadata.type != 'Retrieval']
        oom_tasks = ["AmazonReviewsClassification", "ArxivClusteringS2S", "BiorxivClusteringS2S", "MedrxivClusteringS2S", "MindSmallReranking", "RedditClustering", "SciDocsRR", "StackExchangeClustering", "StackOverflowDupQuestions", "ToxicConversationsClassification", "TwentyNewsgroupsClustering", "TwitterSemEval2015", "TwitterURLCorpus"]
        reduced_mteb_eng_tasks = [task for task in reduced_mteb_eng_tasks if task.metadata.name not in oom_tasks]
    elif args.filter_tasks == 'classification_subset':
        # Specific classification tasks for GNN baseline experiments
        target_tasks = [
            "AmazonCounterfactualClassification",
            "Banking77Classification",
            "EmotionClassification",
            "MTOPDomainClassification",
            "MTOPIntentClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            "PoemSentimentClassification",
            "TweetSentimentExtractionClassification"
        ]
        reduced_mteb_eng_tasks = [task for task in mteb_eng if task.metadata.name in target_tasks]
    else:
        # Custom task list (comma-separated) - load directly by name
        custom_tasks = [t.strip() for t in args.filter_tasks.split(',')]
        # First try to find in benchmark
        reduced_mteb_eng_tasks = [task for task in mteb_eng if task.metadata.name in custom_tasks]
        # If not found in benchmark, try loading directly
        if len(reduced_mteb_eng_tasks) < len(custom_tasks):
            found_names = {task.metadata.name for task in reduced_mteb_eng_tasks}
            missing_tasks = [t for t in custom_tasks if t not in found_names]
            print(f"Tasks not in MTEB(eng) benchmark, loading directly: {missing_tasks}")
            for task_name in missing_tasks:
                try:
                    task = mteb.get_task(task_name)
                    reduced_mteb_eng_tasks.append(task)
                except Exception as e:
                    print(f"  ⚠️ Could not load task '{task_name}': {e}")

    print(f"Running evaluation on {len(reduced_mteb_eng_tasks)} tasks: {[task.metadata.name for task in reduced_mteb_eng_tasks]}")


    # filter tasks that already ran
    if args.purpose == 'run_tasks':
        results_output_folder = f'{args.base_results_path}/{model_family}/{model_size}/{revision}/mteb/layer_{evaluation_layer}'
        if args.use_simple_aggregation:
            results_output_folder += f'_simple_agg_{args.graph_type}_{args.token_pooling_method}_{args.aggregation_method}_{args.nodes_pooling_method}'

        done_tasks = list_done_tasks(results_output_folder)
        reduced_mteb_eng_tasks = [task for task in reduced_mteb_eng_tasks if task.metadata.name not in done_tasks]

    evaluator = mteb.MTEB(tasks=reduced_mteb_eng_tasks)
    
    device_map = "auto" if model_family != 'bert' else None
    
    # Choose model wrapper based on arguments
    if args.use_simple_aggregation:
        model = SimpleAggregationWrapper(
            model_specs, 
            device_map=device_map, 
            evaluation_layer_idx=evaluation_layer,
            aggregation_method=args.aggregation_method,
            token_pooling_method=args.token_pooling_method,
            nodes_pooling_method=args.nodes_pooling_method,
            num_gnn_layers=args.num_gnn_layers,
            graph_type=args.graph_type
        )
        print(f"Using SimpleAggregation wrapper with method={args.aggregation_method}, graph_type={args.graph_type}")
    else:
        model = TextLayerwiseAutoModelWrapper(model_specs, device_map=device_map, evaluation_layer_idx=evaluation_layer)

    # if BERT, we need to manually move the model to the device because the device map is not supported
    # https://github.com/huggingface/transformers/issues/25296
    if model_family == 'bert':
        model.model = model.model.to("cuda:0")

    if args.purpose == 'run_tasks': 
        results_output_folder = f'{args.base_results_path}/{model_family}/{model_size}/{revision}/mteb/layer_{model.evaluation_layer_idx}'
        if args.use_simple_aggregation:
            results_output_folder += f'_simple_agg_{args.graph_type}_{args.token_pooling_method}_{args.aggregation_method}_{args.nodes_pooling_method}'
        
        def custom_create_output_folder(*args):
            output_folder = Path(results_output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            return output_folder
        
        encoding_kwargs = {'verbose': True}
        evaluator.create_output_folder = custom_create_output_folder
        evaluator.run(model, 
                      kwargs=encoding_kwargs, 
                      output_folder='./mteb-results', 
                      raise_error=args.raise_error,
                      overwrite_results=False, 
                      verbosity=2)


if __name__ == "__main__":
    main()
