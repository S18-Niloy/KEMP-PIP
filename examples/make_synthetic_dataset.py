#!/usr/bin/env python
"""
Generate a small synthetic peptide classification dataset.

This is *not* biological data -- it exists purely so that
``examples/run_pipeline.py`` and the package's file I/O can be smoke-tested
without your real dataset. Sequences are drawn from a simple biased random
process: "positive" sequences are enriched for a few residues to give
classifiers something (weak) to learn, so the resulting demo doesn't just
report chance-level metrics.

Usage
-----
    python examples/make_synthetic_dataset.py --out-dir data/synthetic \\
        --n-train 200 --n-test 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
# Residues over-represented in the synthetic "positive" class, loosely
# mimicking cationic/amphipathic antimicrobial-peptide composition.
POSITIVE_ENRICHED = list("KRLW")


def _random_sequence(rng: np.random.Generator, length: int, positive: bool) -> str:
    if positive:
        # 40% chance per residue of drawing from the enriched sub-alphabet.
        pool = [POSITIVE_ENRICHED if rng.random() < 0.4 else AMINO_ACIDS for _ in range(length)]
        return "".join(rng.choice(p) for p in pool)
    return "".join(rng.choice(AMINO_ACIDS, size=length))


def make_split(rng: np.random.Generator, n: int, min_len: int = 8, max_len: int = 40):
    sequences, labels = [], []
    for _ in range(n):
        label = int(rng.random() < 0.5)
        length = int(rng.integers(min_len, max_len + 1))
        sequences.append(_random_sequence(rng, length, positive=bool(label)))
        labels.append(label)
    return sequences, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-test", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_seqs, train_labels = make_split(rng, args.n_train)
    test_seqs, test_labels = make_split(rng, args.n_test)

    def write_col(path: Path, values) -> None:
        with open(path, "w") as f:
            f.write("\n".join(str(v) for v in values) + "\n")

    write_col(out_dir / "X_train.csv", train_seqs)
    write_col(out_dir / "label_train.csv", train_labels)
    write_col(out_dir / "X_test.csv", test_seqs)
    write_col(out_dir / "label_test.csv", test_labels)

    print(f"Wrote synthetic dataset to {out_dir.resolve()}:")
    print(f"  train: {args.n_train} sequences ({sum(train_labels)} positive)")
    print(f"  test:  {args.n_test} sequences ({sum(test_labels)} positive)")


if __name__ == "__main__":
    main()
