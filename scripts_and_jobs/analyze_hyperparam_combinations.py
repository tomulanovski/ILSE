#!/usr/bin/env python3
"""
Analyze which hyperparameter combinations lead to success vs failure.
"""

import argparse
import optuna
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage_url", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="Accuracy threshold for 'success'")
    args = parser.parse_args()

    # Load study
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage_url
    )

    # Get completed trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER COMBINATION ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Total completed trials: {len(completed)}")

    # Create dataframe
    data = []
    for trial in completed:
        row = {
            'trial': trial.number,
            'val_acc': trial.value,
            'success': trial.value >= args.threshold,
            **trial.params
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Count successes vs failures
    success_count = df['success'].sum()
    failure_count = len(df) - success_count

    print(f"Success (val_acc >= {args.threshold}): {success_count} ({100*success_count/len(df):.1f}%)")
    print(f"Failure (val_acc < {args.threshold}):  {failure_count} ({100*failure_count/len(df):.1f}%)")

    # Analyze each hyperparameter
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER IMPACT ON SUCCESS")
    print(f"{'='*80}\n")

    for param in ['dropout', 'lr', 'node_to_choose', 'graph_type', 'gin_layers', 'gin_mlp_layers', 'weight_decay']:
        if param not in df.columns:
            continue

        print(f"\n{param.upper()}:")
        print(f"{'Value':<20} {'Total Trials':<15} {'Successful':<15} {'Success Rate':<15} {'Avg Val Acc':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

        for value in sorted(df[param].unique()):
            subset = df[df[param] == value]
            total = len(subset)
            successful = subset['success'].sum()
            success_rate = 100 * successful / total if total > 0 else 0
            avg_acc = subset['val_acc'].mean()

            # Mark with emoji
            if success_rate > 10:
                marker = "✅"
            elif success_rate > 1:
                marker = "⚠️"
            else:
                marker = "❌"

            print(f"{str(value):<20} {total:<15} {successful:<15} {success_rate:<14.1f}% {avg_acc:<14.4f} {marker}")

    # Find "golden combinations"
    print(f"\n\n{'='*80}")
    print(f"SUCCESSFUL COMBINATIONS (val_acc >= {args.threshold})")
    print(f"{'='*80}\n")

    successful_trials = df[df['success'] == True].sort_values('val_acc', ascending=False)

    if len(successful_trials) == 0:
        print("No successful trials found!")
    else:
        for idx, row in successful_trials.iterrows():
            print(f"\nTrial #{row['trial']} - Val Acc: {row['val_acc']:.4f}")
            print(f"  gin_layers={row['gin_layers']}, dropout={row['dropout']}, gin_mlp_layers={row['gin_mlp_layers']}")
            print(f"  node_to_choose={row['node_to_choose']}, graph_type={row['graph_type']}")
            print(f"  lr={row['lr']}, weight_decay={row['weight_decay']}, batch_size={row['batch_size']}")

    # Probability calculation
    print(f"\n\n{'='*80}")
    print(f"WHY SO FEW SUCCESSFUL TRIALS?")
    print(f"{'='*80}\n")

    # Count configurations
    dropout_values = df['dropout'].nunique()
    lr_values = df['lr'].nunique()
    node_values = df['node_to_choose'].nunique()
    graph_values = df['graph_type'].nunique()
    gin_layer_values = df['gin_layers'].nunique()
    mlp_layer_values = df['gin_mlp_layers'].nunique()

    total_combinations = (dropout_values * lr_values * node_values *
                         graph_values * gin_layer_values * mlp_layer_values)

    print(f"Total possible combinations: {total_combinations}")
    print(f"  dropout: {dropout_values} values")
    print(f"  lr: {lr_values} values")
    print(f"  node_to_choose: {node_values} values")
    print(f"  graph_type: {graph_values} values")
    print(f"  gin_layers: {gin_layer_values} values")
    print(f"  gin_mlp_layers: {mlp_layer_values} values")

    # Calculate how many combinations lead to success
    # Based on the data, estimate which param values work
    successful_params = df[df['success'] == True]
    if len(successful_params) > 0:
        good_dropout = successful_params['dropout'].unique()
        good_lr = successful_params['lr'].unique()
        good_node = successful_params['node_to_choose'].unique()
        good_graph = successful_params['graph_type'].unique()

        prob = (len(good_dropout) / dropout_values *
                len(good_lr) / lr_values *
                len(good_node) / node_values *
                len(good_graph) / graph_values)

        print(f"\nProbability of randomly picking a successful combination:")
        print(f"  Good dropout values: {list(good_dropout)} ({len(good_dropout)}/{dropout_values})")
        print(f"  Good lr values: {list(good_lr)} ({len(good_lr)}/{lr_values})")
        print(f"  Good node_to_choose: {list(good_node)} ({len(good_node)}/{node_values})")
        print(f"  Good graph_type: {list(good_graph)} ({len(good_graph)}/{graph_values})")
        print(f"\n  Combined probability: {prob*100:.1f}%")
        print(f"  Expected successes in {len(df)} trials: {prob*len(df):.1f}")
        print(f"  Actual successes: {success_count}")


if __name__ == "__main__":
    main()