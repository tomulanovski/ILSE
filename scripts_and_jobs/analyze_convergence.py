#!/usr/bin/env python3
"""
Analyze convergence of Optuna trials to determine if training epochs are sufficient.

Usage:
    python scripts_and_jobs/analyze_convergence.py \
        --study_name "gin_MassiveIntentClassification_Pythia_410m" \
        --storage_url "postgresql://user:pass@host:5432/optuna"
"""

import argparse
import optuna
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_convergence(epoch_logs, patience=10):
    """
    Analyze if training converged based on epoch logs.

    Returns:
        dict with convergence metrics
    """
    epochs = [log['epoch'] for log in epoch_logs]
    val_accs = [log['val_acc'] for log in epoch_logs]
    train_accs = [log['train_acc'] for log in epoch_logs]
    val_losses = [log['val_loss'] for log in epoch_logs]
    train_losses = [log['train_loss'] for log in epoch_logs]

    # Find best epoch
    best_epoch_idx = np.argmax(val_accs)
    best_epoch = epochs[best_epoch_idx]
    best_val_acc = val_accs[best_epoch_idx]

    # How many epochs after best epoch?
    epochs_after_best = len(epochs) - best_epoch

    # Check if still improving in last N epochs
    if len(val_accs) >= patience:
        last_n_accs = val_accs[-patience:]
        improving_trend = np.polyfit(range(patience), last_n_accs, deg=1)[0] > 0.0001
    else:
        improving_trend = True

    # Check validation vs training accuracy (overfitting)
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    overfit_gap = final_train_acc - final_val_acc

    # Convergence decision
    converged = (
        epochs_after_best >= patience and  # Best epoch was at least 'patience' epochs ago
        not improving_trend                # Not improving in last 'patience' epochs
    )

    needs_more_epochs = (
        best_epoch >= len(epochs) - 5 or   # Best epoch in last 5 epochs
        improving_trend                     # Still improving
    )

    return {
        'total_epochs': len(epochs),
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'final_val_acc': final_val_acc,
        'epochs_after_best': epochs_after_best,
        'converged': converged,
        'needs_more_epochs': needs_more_epochs,
        'still_improving': improving_trend,
        'overfit_gap': overfit_gap,
        'overfitting': overfit_gap > 0.1,
        'val_accs': val_accs,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'train_losses': train_losses,
        'epochs': epochs
    }


def print_convergence_report(trial, analysis, trial_num=None):
    """Print human-readable convergence report."""
    print("\n" + "="*70)
    if trial_num is not None:
        print(f"CONVERGENCE ANALYSIS - Trial #{trial_num}")
    else:
        print(f"CONVERGENCE ANALYSIS")
    print("="*70)

    print(f"\n📊 Training Summary:")
    print(f"  Total epochs:        {analysis['total_epochs']}")
    print(f"  Best epoch:          {analysis['best_epoch']}")
    print(f"  Best val accuracy:   {analysis['best_val_acc']:.4f}")
    print(f"  Final val accuracy:  {analysis['final_val_acc']:.4f}")
    print(f"  Epochs after best:   {analysis['epochs_after_best']}")

    print(f"\n🎯 Convergence Status:")
    if analysis['converged']:
        print(f"  ✅ CONVERGED - Training stopped improving {analysis['epochs_after_best']} epochs ago")
        print(f"  👍 50 epochs is ENOUGH (or even too many)")
        if analysis['epochs_after_best'] > 20:
            print(f"  💡 Could use early stopping with patience={analysis['epochs_after_best']//2}")
    elif analysis['needs_more_epochs']:
        print(f"  ⚠️  NOT CONVERGED - Still improving or best epoch too recent")
        print(f"  📈 Recommend: INCREASE to 75-100 epochs")
    else:
        print(f"  ✅ CONVERGED - Training appears stable")
        print(f"  👍 50 epochs is APPROPRIATE")

    print(f"\n🔍 Additional Metrics:")
    print(f"  Still improving:     {'Yes ⬆️' if analysis['still_improving'] else 'No ➡️'}")
    print(f"  Train-val gap:       {analysis['overfit_gap']:.4f}")
    if analysis['overfitting']:
        print(f"  ⚠️  OVERFITTING DETECTED (gap > 0.1)")
        print(f"  💡 Consider: more dropout, weight decay, or early stopping")
    else:
        print(f"  ✅ No significant overfitting")

    # Show last 10 epochs
    print(f"\n📉 Last 10 Epochs:")
    print(f"  {'Epoch':>6} {'Train Acc':>10} {'Val Acc':>10} {'Val Loss':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(max(0, len(analysis['epochs'])-10), len(analysis['epochs'])):
        epoch = analysis['epochs'][i]
        train_acc = analysis['train_accs'][i]
        val_acc = analysis['val_accs'][i]
        val_loss = analysis['val_losses'][i]
        marker = " 🌟" if epoch == analysis['best_epoch'] else ""
        print(f"  {epoch:>6} {train_acc:>10.4f} {val_acc:>10.4f} {val_loss:>10.4f}{marker}")

    print("="*70 + "\n")


def plot_training_curves(trial, analysis, output_path=None):
    """Plot training curves to visualize convergence."""
    epochs = analysis['epochs']
    train_accs = analysis['train_accs']
    val_accs = analysis['val_accs']
    val_losses = analysis['val_losses']
    train_losses = analysis['train_losses']
    best_epoch = analysis['best_epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax1.axvline(best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training Progress - Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot train and validation loss
    ax2.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
    ax2.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    ax2.axvline(best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Progress - Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training curve saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze convergence of Optuna trials")
    parser.add_argument("--study_name", type=str, required=True, help="Optuna study name")
    parser.add_argument("--storage_url", type=str, required=True, help="PostgreSQL storage URL")
    parser.add_argument("--trial_number", type=int, default=None, help="Specific trial to analyze (default: best trial)")
    parser.add_argument("--analyze_all", action="store_true", help="Analyze all completed trials")
    parser.add_argument("--plot", action="store_true", help="Generate training curve plots")
    parser.add_argument("--output_dir", type=str, default="./convergence_analysis", help="Output directory for plots")
    args = parser.parse_args()

    # Create output directory
    if args.plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load study
    print(f"Loading study: {args.study_name}")
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage_url
    )

    print(f"Study has {len(study.trials)} trials")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Completed trials: {len(completed_trials)}")

    if not completed_trials:
        print("❌ No completed trials found!")
        return

    if args.analyze_all:
        # Analyze all trials
        print(f"\nAnalyzing all {len(completed_trials)} completed trials...")
        convergence_stats = []

        for trial in completed_trials:
            if 'epoch_logs' not in trial.user_attrs:
                continue

            epoch_logs = trial.user_attrs['epoch_logs']
            analysis = analyze_convergence(epoch_logs)
            convergence_stats.append({
                'trial_number': trial.number,
                'best_epoch': analysis['best_epoch'],
                'converged': analysis['converged'],
                'needs_more_epochs': analysis['needs_more_epochs']
            })

        # Summary statistics
        avg_best_epoch = np.mean([s['best_epoch'] for s in convergence_stats])
        pct_converged = 100 * sum(s['converged'] for s in convergence_stats) / len(convergence_stats)
        pct_needs_more = 100 * sum(s['needs_more_epochs'] for s in convergence_stats) / len(convergence_stats)

        print("\n" + "="*70)
        print("SUMMARY ACROSS ALL TRIALS")
        print("="*70)
        print(f"Average best epoch:          {avg_best_epoch:.1f}")
        print(f"Trials that converged:       {pct_converged:.1f}%")
        print(f"Trials needing more epochs:  {pct_needs_more:.1f}%")
        print("="*70)

        if pct_needs_more > 30:
            print(f"\n⚠️  RECOMMENDATION: Increase epochs to 75-100")
        elif avg_best_epoch < 25:
            print(f"\n💡 RECOMMENDATION: Could reduce epochs to 30-40 with early stopping")
        else:
            print(f"\n✅ RECOMMENDATION: 50 epochs is appropriate")

    else:
        # Analyze specific trial
        if args.trial_number is not None:
            trial = study.trials[args.trial_number]
            trial_desc = f"Trial #{args.trial_number}"
        else:
            trial = study.best_trial
            trial_desc = f"Best Trial (#{trial.number})"

        print(f"\nAnalyzing {trial_desc}...")
        print(f"Trial value: {trial.value:.4f}")
        print(f"Hyperparameters: {trial.params}")

        if 'epoch_logs' not in trial.user_attrs:
            print("❌ No epoch logs found for this trial!")
            return

        epoch_logs = trial.user_attrs['epoch_logs']
        analysis = analyze_convergence(epoch_logs)

        # Print report
        print_convergence_report(trial, analysis, trial.number)

        # Plot if requested
        if args.plot:
            plot_path = output_dir / f"trial_{trial.number}_training_curves.png"
            plot_training_curves(trial, analysis, output_path=plot_path)


if __name__ == "__main__":
    main()