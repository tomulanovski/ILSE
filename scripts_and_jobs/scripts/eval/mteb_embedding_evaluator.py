#!/usr/bin/env python3
"""
Generic MTEB-style embedding evaluator.
Takes embeddings and evaluates using MTEB evaluation logic.
Matches MTEB library's exact behavior and output format.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from collections import defaultdict


def evaluate_classification_embeddings(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    task_name: str = "",
    n_experiments: int = 10,
    samples_per_label: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate embeddings on classification task using MTEB's EXACT methodology.

    Matches MTEB library behavior precisely:
    - Uses same seed for all experiments (MTEB behavior)
    - Shuffles same idxs list repeatedly (accumulating shuffles)
    - Computes all MTEB metrics per experiment
    - Returns FullClassificationMetrics format

    Args:
        train_embeddings: (N_train, D) training embeddings
        train_labels: (N_train,) training labels
        test_embeddings: (N_test, D) test embeddings
        test_labels: (N_test,) test labels
        task_name: Name of the task (optional)
        n_experiments: Number of experiments (default: 10, matches MTEB)
        samples_per_label: Samples per class for undersampling (default: 8, matches MTEB)
        seed: Random seed (default: 42, matches MTEB)

    Returns:
        Dict with FullClassificationMetrics format:
        - All metrics averaged across experiments
        - scores_per_experiment: List of per-experiment results
    """
    print(f"Running experiment (0/{n_experiments})")

    # Determine if binary classification (for ap scores)
    num_classes = len(np.unique(test_labels))
    is_binary = (num_classes == 2)

    # Run multiple experiments - MTEB style
    scores_per_experiment: List[Dict[str, Any]] = []

    # Create index array - will be shuffled in-place repeatedly (MTEB behavior)
    idxs = list(range(len(train_embeddings)))

    for exp_idx in range(n_experiments):
        if exp_idx > 0:
            print(f"Running experiment ({exp_idx}/{n_experiments})")

        # MTEB's undersampling: Use SAME seed, shuffle same idxs list repeatedly
        # This matches MTEB's exact behavior in _undersample_data
        rng_state = np.random.RandomState(seed)
        rng_state.shuffle(idxs)  # Shuffle in-place (idxs accumulates shuffles!)

        # Select samples by iterating through shuffled indices
        label_counter = defaultdict(int)
        sampled_indices = []

        for idx in idxs:
            label = train_labels[idx]
            if label_counter[label] < samples_per_label:
                sampled_indices.append(idx)
                label_counter[label] += 1

        # Create balanced training set
        train_emb_balanced = train_embeddings[sampled_indices]
        train_labels_balanced = train_labels[sampled_indices]

        # Train classifier (match MTEB parameters exactly)
        clf = LogisticRegression(n_jobs=-1, max_iter=100, random_state=seed)
        clf.fit(train_emb_balanced, train_labels_balanced)

        # Predict on test set
        y_pred = clf.predict(test_embeddings)

        # Compute all MTEB metrics (matches _calculate_scores)
        exp_scores = {
            'accuracy': float(accuracy_score(test_labels, y_pred)),
            'f1': float(f1_score(test_labels, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(test_labels, y_pred, average='weighted', zero_division=0)),
            'precision': float(precision_score(test_labels, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(test_labels, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(test_labels, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(test_labels, y_pred, average='weighted', zero_division=0)),
            'ap': None,
            'ap_weighted': None
        }

        # Compute average precision for binary classification only
        if is_binary:
            exp_scores['ap'] = float(average_precision_score(test_labels, y_pred, average='macro'))
            exp_scores['ap_weighted'] = float(average_precision_score(test_labels, y_pred, average='weighted'))

        scores_per_experiment.append(exp_scores)

    # Compute averages across experiments (matches MTEB's aggregation logic)
    # For each metric, compute mean of non-None values
    avg_scores = {}
    for metric_key in scores_per_experiment[0].keys():
        values = [s[metric_key] for s in scores_per_experiment if s[metric_key] is not None]
        if values:
            avg_scores[metric_key] = float(np.mean(values))
        else:
            avg_scores[metric_key] = np.nan

    print(f"Running {task_name} - Finished.")

    # Return FullClassificationMetrics format (matches MTEB exactly)
    results = {
        # Averaged metrics (same as ClassificationMetrics)
        'accuracy': avg_scores['accuracy'],
        'f1': avg_scores['f1'],
        'f1_weighted': avg_scores['f1_weighted'],
        'precision': avg_scores['precision'],
        'precision_weighted': avg_scores['precision_weighted'],
        'recall': avg_scores['recall'],
        'recall_weighted': avg_scores['recall_weighted'],
        'ap': avg_scores['ap'],
        'ap_weighted': avg_scores['ap_weighted'],
        # Full metrics
        'scores_per_experiment': scores_per_experiment
    }

    return results


def evaluate_sts_embeddings(
    test_embeddings_a: np.ndarray,
    test_embeddings_b: np.ndarray,
    test_scores: np.ndarray,
    task_name: str,
    similarity_metric: str = "cosine"
) -> Dict[str, Any]:
    """
    Evaluate embeddings on STS (Semantic Textual Similarity) task.

    Args:
        test_embeddings_a: (N, D) embeddings for sentence A
        test_embeddings_b: (N, D) embeddings for sentence B
        test_scores: (N,) ground truth similarity scores
        task_name: Name of the task
        similarity_metric: 'cosine' or 'dot'

    Returns:
        Dict with spearman correlation
    """
    print(f"Computing {similarity_metric} similarities for {len(test_embeddings_a)} pairs...")

    # Compute similarities
    if similarity_metric == "cosine":
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.array([
            cosine_similarity(test_embeddings_a[i:i+1], test_embeddings_b[i:i+1])[0, 0]
            for i in range(len(test_embeddings_a))
        ])
    elif similarity_metric == "dot":
        # Dot product similarity
        similarities = np.sum(test_embeddings_a * test_embeddings_b, axis=1)
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    # Compute Spearman correlation
    spearman_corr, _ = spearmanr(test_scores, similarities)

    results = {
        'task_name': task_name,
        'task_type': 'sts',
        'spearman': float(spearman_corr),
        'similarity_metric': similarity_metric,
        'test_samples': len(test_scores)
    }

    return results


def evaluate_embeddings(
    task_type: str,
    task_name: str,
    train_embeddings: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    test_embeddings: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    test_embeddings_a: Optional[np.ndarray] = None,
    test_embeddings_b: Optional[np.ndarray] = None,
    test_scores: Optional[np.ndarray] = None,
    similarity_metric: str = "cosine"
) -> Dict[str, Any]:
    """
    Generic embedding evaluation dispatcher.

    Args:
        task_type: 'classification' or 'sts'
        task_name: Name of the task

        For classification:
            train_embeddings: (N_train, D)
            train_labels: (N_train,)
            test_embeddings: (N_test, D)
            test_labels: (N_test,)

        For STS:
            test_embeddings_a: (N, D)
            test_embeddings_b: (N, D)
            test_scores: (N,)
            similarity_metric: 'cosine' or 'dot'

    Returns:
        Dict with evaluation metrics
    """
    if task_type == "classification":
        if train_embeddings is None or test_embeddings is None:
            raise ValueError("Classification requires train_embeddings and test_embeddings")

        return evaluate_classification_embeddings(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            task_name=task_name
        )

    elif task_type == "sts":
        if test_embeddings_a is None or test_embeddings_b is None:
            raise ValueError("STS requires test_embeddings_a and test_embeddings_b")

        return evaluate_sts_embeddings(
            test_embeddings_a=test_embeddings_a,
            test_embeddings_b=test_embeddings_b,
            test_scores=test_scores,
            task_name=task_name,
            similarity_metric=similarity_metric
        )

    else:
        raise ValueError(f"Unknown task type: {task_type}")
