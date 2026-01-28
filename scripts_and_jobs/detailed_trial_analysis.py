#!/usr/bin/env python3
"""
Detailed analysis of individual trials to understand training dynamics.
"""

import argparse
import optuna
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage_url", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=5, help="Number of top trials to analyze")
    args = parser.parse_args()

    # Load study
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage_url
    )

    # Get top N trials by validation accuracy
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed = sorted(completed, key=lambda t: t.value, reverse=True)[:args.num_trials]

    print(f"\n{'='*80}")
    print(f"TOP {args.num_trials} TRIALS - DETAILED ANALYSIS")
    print(f"{'='*80}\n")

    for rank, trial in enumerate(completed, 1):
        print(f"\n{'='*80}")
        print(f"RANK #{rank} - Trial #{trial.number}")
        print(f"{'='*80}")
        print(f"Best Val Accuracy: {trial.value:.4f}")
        print(f"Parameters: {trial.params}")

        if 'epoch_logs' not in trial.user_attrs:
            print("  ⚠️  No epoch logs available")
            continue

        epoch_logs = trial.user_attrs['epoch_logs']
        best_epoch = trial.user_attrs.get('best_epoch', 'N/A')

        print(f"\nBest Epoch: {best_epoch}/{len(epoch_logs)}")
        print(f"\n{'Epoch':>6} {'Train Acc':>12} {'Val Acc':>12} {'Val Loss':>12} {'Notes':>20}")
        print(f"{'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")

        for i, log in enumerate(epoch_logs):
            epoch = log['epoch']
            train_acc = log['train_acc']
            val_acc = log['val_acc']
            val_loss = log['val_loss']

            notes = []
            if epoch == best_epoch:
                notes.append("🌟 BEST")
            if i > 0:
                prev_val = epoch_logs[i-1]['val_acc']
                if val_acc < prev_val - 0.001:
                    notes.append("⬇️ DROP")
                elif val_acc > prev_val + 0.001:
                    notes.append("⬆️ IMPROVE")

            note_str = " ".join(notes) if notes else ""
            print(f"{epoch:>6} {train_acc:>12.4f} {val_acc:>12.4f} {val_loss:>12.4f} {note_str:>20}")

        # Calculate improvement trajectory
        val_accs = [log['val_acc'] for log in epoch_logs]
        first_5_avg = np.mean(val_accs[:5]) if len(val_accs) >= 5 else val_accs[0]
        last_5_avg = np.mean(val_accs[-5:]) if len(val_accs) >= 5 else val_accs[-1]
        improvement = last_5_avg - first_5_avg

        print(f"\n📈 Training Dynamics:")
        print(f"  First 5 epochs avg:  {first_5_avg:.4f}")
        print(f"  Last 5 epochs avg:   {last_5_avg:.4f}")
        print(f"  Total improvement:   {improvement:+.4f}")

        # Check if random or learning
        if trial.value < 0.02:
            print(f"  ⚠️  WARNING: Val accuracy very low - model might not be learning!")
        elif best_epoch <= 3:
            print(f"  ⚠️  WARNING: Best epoch very early - might be random initialization luck")
        elif improvement < 0.001:
            print(f"  ⚠️  WARNING: No improvement from start to end - model not learning")

    # Overall statistics
    print(f"\n\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")

    all_completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    val_accs = [t.value for t in all_completed]
    best_epochs = [t.user_attrs.get('best_epoch', 50) for t in all_completed if 'best_epoch' in t.user_attrs]

    print(f"Total completed trials:     {len(all_completed)}")
    print(f"Best val accuracy:          {max(val_accs):.4f}")
    print(f"Median val accuracy:        {np.median(val_accs):.4f}")
    print(f"Worst val accuracy:         {min(val_accs):.4f}")
    print(f"Avg best epoch:             {np.mean(best_epochs):.1f}")
    print(f"Median best epoch:          {np.median(best_epochs):.0f}")

    # Check if models are actually learning
    poor_trials = sum(1 for v in val_accs if v < 0.02)
    if poor_trials > len(all_completed) * 0.3:
        print(f"\n⚠️  WARNING: {poor_trials}/{len(all_completed)} trials have very low accuracy (<2%)")
        print(f"    This suggests a potential issue with:")
        print(f"    - Learning rate too high/low")
        print(f"    - Model architecture")
        print(f"    - Data preprocessing")
        print(f"    - Number of classes (are labels correct?)")


if __name__ == "__main__":
    main()