import os
import pickle
from typing import Any, Callable, List, Literal, Type, Dict, Union
import json
import mteb
import pandas as pd
import numpy as np
from ..metrics.metric_calling import EvaluationMetricSpecifications
from ..model_definitions.base_automodel_wrapper import BaseModelSpecifications

def construct_file_path(
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any],
        base_path: str = "experiments/results",
        include_split: bool = False
):
    model_family = model_specs.model_family
    model_size = model_specs.model_size
    revision = model_specs.revision
    evaluation_metric = evaluation_metric_specs.evaluation_metric
    granularity = evaluation_metric_specs.granularity
    dataset = dataloader_kwargs['dataset_name']

    if evaluation_metric == 'entropy':
        evaluation_metric = f"{evaluation_metric}_{granularity}"

    if include_split:
        split = dataloader_kwargs['split']
        return f"{base_path}/{model_family}/{model_size}/{revision}/metrics/{dataset}/{split}/{evaluation_metric}.pkl"
    else:
        return f"{base_path}/{model_family}/{model_size}/{revision}/metrics/{dataset}/{evaluation_metric}.pkl"

def save_results(
        results, 
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any]
):
    file_path = construct_file_path(model_specs, evaluation_metric_specs, dataloader_kwargs, include_split=True)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(results, f)

def check_if_results_exist(
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any]
):
    file_path = construct_file_path(model_specs, evaluation_metric_specs, dataloader_kwargs)
    return os.path.exists(file_path)


def load_results(
        model_specs: BaseModelSpecifications, 
        evaluation_metric_specs: EvaluationMetricSpecifications, 
        dataloader_kwargs: Dict[str, Any],
        base_path: str = 'experiments/results'
):
    file_path = construct_file_path(model_specs, evaluation_metric_specs, dataloader_kwargs, base_path=base_path)

    try:
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def load_results_for_model_and_revisions(model_family, model_size, revisions, evaluation_metrics, base_path='experiments/results'):
    results = {}
    for revision in revisions:
        model_specs = BaseModelSpecifications(model_family, model_size, revision, ignore_checks=True)
        for evaluation_metric in evaluation_metrics:
            evaluation_metric_specs = EvaluationMetricSpecifications(evaluation_metric)
            dataloader_kwargs = {'dataset_name': 'wikitext/train'}
            results[(revision, evaluation_metric)] = load_results(model_specs, evaluation_metric_specs, dataloader_kwargs, base_path=base_path)
    return results

def adjust_infonce_scores(result, model_family):
    dimensionalities = {
        'bert': 768,
        'roberta': 768,
        'mamba': 1024,
        'Pythia': 1024
    }
    upper_bound = np.log(dimensionalities[model_family])

    return 1 - result / upper_bound
    

def load_all_results(
        should_normalize_scores_across_models: bool = False,
        base_path: str = 'experiments/results'
):
    """
    This is ugly code but it works. Basically it expects the folder structure to be as follows:


    experiments/results/
        model_family/
            model_size/
                revision/
                    metrics/
                        dataset_name/
                            evaluation_metric.pkl
                            ...
                    mteb/
                        layer_number/
                            dataset_name.json
                            ...

    Arguments:
        should_normalize_scores_across_models: If True, normalize scores to have the same mean and std at the model-revision-layer level.
    """
    all_results = {}
    results_dir = base_path


    # for each task, find the corresponding dataset name  
    mteb_eng = mteb.get_benchmark("MTEB(eng)")
    reduced_mteb_eng_tasks = [task for task in mteb_eng if task.metadata.category != 'p2p']
    reduced_mteb_eng_tasks = [task for task in reduced_mteb_eng_tasks if task.metadata.type != 'Retrieval']
    evaluator = mteb.MTEB(tasks=reduced_mteb_eng_tasks)
    task_name_to_dataset_name = {}
    for task in evaluator.tasks:
        task_name_to_dataset_name[task.metadata.name] = task.metadata.dataset['path'].split('/')[1]
    dataset_name_to_task_name = {v: k for k, v in task_name_to_dataset_name.items()}

    # iterate over model families
    vision_model_families = ['dinov2', 'dinov2-register', 'mae', 'clip', 'vit', 'i-jepa']
    for model_family in os.listdir(results_dir):
        if model_family in vision_model_families:
            continue

        model_family_path = os.path.join(results_dir, model_family)
        if os.path.isdir(model_family_path):
            all_results[model_family] = {}
            
            # iterate over model sizes
            for model_size in os.listdir(model_family_path):
                model_size_path = os.path.join(model_family_path, model_size)
                if os.path.isdir(model_size_path):
                    all_results[model_family][model_size] = {}
                    
                    # iterate over revisions
                    for revision in os.listdir(model_size_path):
                        revision_path = os.path.join(model_size_path, revision)
                        if os.path.isdir(revision_path):
                            all_results[model_family][model_size][revision] = {}
                            

                            # load MTEB results
                            mteb_path = os.path.join(revision_path, "mteb")
                            if os.path.isdir(mteb_path):
                                all_results[model_family][model_size][revision] = {}
                                for layer in os.listdir(mteb_path):
                                    layer_path = os.path.join(mteb_path, layer)
                                    if os.path.isdir(layer_path):
                                        all_results[model_family][model_size][revision][layer] = {}
                                        for dataset_file in os.listdir(layer_path):
                                            if dataset_file.endswith('.json') and 'model_meta' not in dataset_file:
                                                file_path = os.path.join(layer_path, dataset_file)
                                                with open(file_path, 'r') as f:
                                                    task_results = json.load(f)


                                                task_name = task_results['task_name']
                                                main_score = task_results['scores']['test'][0]['main_score']
                                                all_results[model_family][model_size][revision][layer][task_name] = {
                                                    'main_score': main_score,
                                                    'dataset_name': task_name_to_dataset_name[task_name]
                                                }

                                                    
                                # sort the layer keys 
                                all_results[model_family][model_size][revision] = dict(sorted(all_results[model_family][model_size][revision].items(), key=lambda x: int(x[0].split('_')[1])))



                            # load metrics
                            metrics_path = os.path.join(revision_path, "metrics", "mteb")
                            
                            if os.path.isdir(metrics_path):
                                for dataset in os.listdir(metrics_path):
                                    dataset_path = os.path.join(metrics_path,  dataset, "test")
                                    if os.path.isdir(dataset_path):
                                        
                                        for metric_file in os.listdir(dataset_path):
                                            if metric_file.endswith('.pkl'):
                                                metric_name = os.path.splitext(metric_file)[0]
                                                file_path = os.path.join(dataset_path, metric_file)
                                                with open(file_path, 'rb') as f:
                                                    metric_results = pickle.load(f)

                                                # each metric pkl is a dictionary of lists of floats, one float per layer
                                                # we need to put this into the already existing all_results structure
                                                for metric_normalization, metric_values in metric_results.items():
                                                    for layer, metric_value in enumerate(metric_values):
                                                        correct_task_name = dataset_name_to_task_name[dataset]
                                                        if dataset != 'wikitext':
                                                            try:
                                                                if not metric_name in all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name]:
                                                                    all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name][metric_name] = {}
                                                                all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name][metric_name][metric_normalization] = metric_value

                                                                if metric_name == 'infonce':
                                                                    all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name][metric_name]['fixed'] = adjust_infonce_scores(metric_value, model_family)
                                                            except Exception as e:
                                                                #print(f"Error: {e}", model_family, model_size, revision, f"layer_{layer}", correct_task_name, metric_name, metric_normalization, metric_value)
                                                                pass

                            # load wikitext metrics
                            metrics_path = os.path.join(revision_path, "metrics", "wikitext/train")

                            if os.path.isdir(metrics_path):
                                for metric_file in os.listdir(metrics_path):
                                    if metric_file.endswith('.pkl'):
                                        metric_name = os.path.splitext(metric_file)[0]
                                        file_path = os.path.join(metrics_path, metric_file)
                                        with open(file_path, 'rb') as f:
                                            metric_results = pickle.load(f)

                                        # each metric pkl is a dictionary of lists of floats, one float per layer
                                        # we need to put this into the already existing all_results structure
                                        for metric_normalization, metric_values in metric_results.items():
                                            for layer, metric_value in enumerate(metric_values):
                                                correct_task_name = 'wikitext'
                                                try:
                                                    if f"layer_{layer}" not in all_results[model_family][model_size][revision]:
                                                        all_results[model_family][model_size][revision][f"layer_{layer}"] = {}
                                                    if not correct_task_name in all_results[model_family][model_size][revision][f"layer_{layer}"]:
                                                        all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name] = {}

                                                    if not metric_name in all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name]:
                                                        all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name][metric_name] = {}
                                                    all_results[model_family][model_size][revision][f"layer_{layer}"][correct_task_name][metric_name][metric_normalization] = metric_value
                                                
                                                except Exception as e:
                                                    raise e
                                                    # print(f"Error: {e}", model_family, model_size, revision, f"layer_{layer}", correct_task_name, metric_name, metric_normalization, metric_value)
                                                    pass

    if should_normalize_scores_across_models:
        for model_family in all_results:
            for model_size in all_results[model_family]:
                for revision in all_results[model_family][model_size]:
                    for layer in all_results[model_family][model_size][revision]:
                        main_scores = [all_results[model_family][model_size][revision][layer][task_name]['main_score'] for task_name in all_results[model_family][model_size][revision][layer]]
                        mean_main_score = np.mean(main_scores)
                        std_main_score = np.std(main_scores)

                        entropy_scores = [all_results[model_family][model_size][revision][layer][task_name]['entropy_sentence']['maxEntropy'] for task_name in all_results[model_family][model_size][revision][layer] if 'entropy_sentence' in all_results[model_family][model_size][revision][layer][task_name]]
                        mean_entropy_score = np.mean(entropy_scores)
                        std_entropy_score = np.std(entropy_scores)

                        print(mean_entropy_score, std_entropy_score)    

                        for task_name in all_results[model_family][model_size][revision][layer]:
                            if task_name == 'wikitext':
                                continue

                            all_results[model_family][model_size][revision][layer][task_name]['standardized_main_score'] = (all_results[model_family][model_size][revision][layer][task_name]['main_score'] - mean_main_score) / std_main_score

                            if 'entropy_sentence' in all_results[model_family][model_size][revision][layer][task_name]:
                                all_results[model_family][model_size][revision][layer][task_name]['entropy_sentence']['standardized_logD'] = (all_results[model_family][model_size][revision][layer][task_name]['entropy_sentence']['maxEntropy'] - mean_entropy_score) / std_entropy_score

    return all_results
