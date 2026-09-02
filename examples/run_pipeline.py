#!/usr/bin/env python
"""
End-to-end reproducible example for the peptide_pipeline package.

This script reproduces the full notebook workflow from the command line:

  1. load train/test peptide sequences + labels from CSV
  2. extract k-mer, physchem, modlAMP, and ESM-2 features
  3. run the 5-combo (A-E) coefficient-pruned Logistic Regression ensemble
     ablation, with pairwise alpha-blend ensembling
  4. run the 15-combo x 4-model (XGB/RF/SVM/MLP) feature-block ablation
  5. (optional) benchmark training time for every combo/model pair
  6. save all leaderboards, ROC plots, and coefficient-interpretation
     tables to --out-dir

Usage
-----
    python examples/run_pipeline.py \\
        --train-sequences data/X_train.csv \\
        --train-labels data/label_train.csv \\
        --test-sequences data/X_test.csv \\
        --test-labels data/label_test.csv \\
        --out-dir results/

Test labels are optional -- omit --test-labels to run in predict-only mode.
See --help for all options, including --skip-multi-model-ablation and
--skip-training-time to shorten the run for a first smoke test (ESM-2
feature extraction and the 15-combo x 4-model ablation are the slowest
steps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this script directly from the `examples/` directory without
# installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peptide_pipeline.ablation import prepare_ablation_blocks, run_block_ablation
from peptide_pipeline.data_loading import load_train_test
from peptide_pipeline.ensemble import run_ensemble_ablation
from peptide_pipeline.interpretation import build_coefficient_table, summarize_positive_quadrant
from peptide_pipeline.pipeline import compute_class_weights, extract_all_features
from peptide_pipeline.plotting import plot_ensemble_rocs_separately
from peptide_pipeline.training_time import measure_training_time, summarize_training_time
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-sequences", required=True, help="Header-less CSV, one peptide sequence per row.")
    p.add_argument("--train-labels", required=True, help="Header-less CSV, one 0/1 label per row (aligned to --train-sequences).")
    p.add_argument("--test-sequences", required=True, help="Header-less CSV, one peptide sequence per row.")
    p.add_argument("--test-labels", default=None, help="Optional header-less CSV of 0/1 test labels. Omit for predict-only mode.")
    p.add_argument("--out-dir", default="results", help="Directory to write CSVs and figures to (created if missing).")
    p.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D", help="HuggingFace ESM-2 checkpoint to use.")
    p.add_argument("--esm-batch-size", type=int, default=16)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--decision-threshold", type=float, default=0.4, help="Fixed threshold used during ablation scoring.")
    p.add_argument("--skip-multi-model-ablation", action="store_true", help="Skip the slower 15-combo x 4-model (XGB/RF/SVM/MLP) ablation.")
    p.add_argument("--skip-training-time", action="store_true", help="Skip the training-time benchmarking step.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load data
    # ------------------------------------------------------------------
    print("=" * 80)
    print("1) Loading data")
    print("=" * 80)
    train_data, test_data = load_train_test(
        args.train_sequences, args.train_labels, args.test_sequences, args.test_labels
    )
    X_seq_train = train_data["peptide_sequence"].tolist()
    X_seq_test = test_data["peptide_sequence"].tolist()
    y_train = train_data["label"].values.astype(int)
    y_test = test_data["label"].values.astype(int) if "label" in test_data.columns else None
    class_weight_dict = compute_class_weights(y_train)

    # ------------------------------------------------------------------
    # 2) Feature extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2) Extracting features")
    print("=" * 80)
    feature_set = extract_all_features(
        X_seq_train, X_seq_test,
        esm_model_name=args.esm_model,
        esm_batch_size=args.esm_batch_size,
    )

    # ------------------------------------------------------------------
    # 3) Ensemble ablation (5 combos A-E, coefficient-pruned LR)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3) Ensemble ablation (A-E combos, coefficient-pruned Logistic Regression)")
    print("=" * 80)
    models, leaderboard = run_ensemble_ablation(
        feature_set.as_block_tuples_with_names(),
        y_train, class_weight_dict, y_test=y_test,
        val_size=args.val_size, random_state=args.random_state,
    )
    if not leaderboard.empty:
        leaderboard.to_csv(out_dir / "ensemble_leaderboard.csv", index=False)
        print("\nSaved ensemble_leaderboard.csv")
        print(leaderboard.to_string(index=False))

        # ROC curves for the pairwise ensembles
        plot_ensemble_rocs_separately(models, y_test, out_dir=str(out_dir / "roc_figures"))

        # Coefficient interpretation table for each single combo model
        interp_dir = out_dir / "interpretation"
        interp_dir.mkdir(exist_ok=True)
        for combo_name, info in models.items():
            coef = info["final_model"].coef_.ravel()
            table = build_coefficient_table(info["kept_names"], info["kept_groups"], coef)
            table.to_csv(interp_dir / f"{combo_name}_coefficients.csv", index=False)
            summarize_positive_quadrant(table, name=combo_name)
    else:
        print("No test labels provided -- per-combo predictions were not scored (extend this script to save predictions_*.csv if needed).")

    # ------------------------------------------------------------------
    # 4) Multi-model feature-block ablation (15 combos x XGB/RF/SVM/MLP)
    # ------------------------------------------------------------------
    if not args.skip_multi_model_ablation:
        print("\n" + "=" * 80)
        print("4) Multi-model feature-block ablation (15 combos x XGB/RF/SVM/MLP)")
        print("=" * 80)
        feature_blocks_raw = {
            name: (feature_set.train[name], feature_set.test[name]) for name in feature_set.train
        }
        val_df, test_df = run_block_ablation(
            feature_blocks_raw, y_train, y_test=y_test,
            val_size=args.val_size, random_state=args.random_state,
            decision_threshold=args.decision_threshold,
        )
        val_df.to_csv(out_dir / "ablation_validation_summary.csv", index=False)
        print("\nSaved ablation_validation_summary.csv")
        print(val_df.head(10).to_string(index=False))
        if test_df is not None:
            test_df.to_csv(out_dir / "ablation_test_summary.csv", index=False)
            print("\nSaved ablation_test_summary.csv")
            print(test_df.head(10).to_string(index=False))

        # --------------------------------------------------------------
        # 5) Training-time benchmarking (reuses the same blocks/combos)
        # --------------------------------------------------------------
        if not args.skip_training_time:
            print("\n" + "=" * 80)
            print("5) Training-time benchmarking")
            print("=" * 80)
            blocks_raw_tr, blocks_raw_te, blocks_scl_tr, blocks_scl_te = prepare_ablation_blocks(feature_blocks_raw)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.random_state)
            idx_tr, idx_val = next(sss.split(blocks_raw_tr["kmer"], y_train))

            timing_df = measure_training_time(
                blocks_raw_tr, blocks_scl_tr, blocks_raw_te, blocks_scl_te,
                y_train, idx_tr, idx_val=idx_val, y_test=y_test,
                random_state=args.random_state, include_test=y_test is not None,
            )
            timing_df.to_csv(out_dir / "ablation_training_time_detailed.csv", index=False)

            val_pivot, test_pivot, model_summary = summarize_training_time(timing_df)
            val_pivot.to_csv(out_dir / "ablation_training_time_validation_pivot.csv", index=False)
            if not test_pivot.empty:
                test_pivot.to_csv(out_dir / "ablation_training_time_test_pivot.csv", index=False)
            model_summary.to_csv(out_dir / "ablation_training_time_model_summary.csv", index=False)
            print("\nSaved training-time CSVs")
            print(model_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Done. All outputs written under: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
