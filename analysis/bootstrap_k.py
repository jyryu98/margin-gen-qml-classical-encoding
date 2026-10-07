"""
bootstrap_k.py

K(mu) bootstrap.

This resamples the five random-seed blocks with replacement, relabels them back to the canonical
seed values, recomputes K(mu) over all CI conditioning sets, and writes JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

import extra_measures_test as E


SEED_VALUES = [16, 13, 42, 65, 89]

DEFAULT_K_NAMES = [
    "sqrt(nq)/mm",
    "1/mm",
    "np/mm",
    "np*sqrt(nq)",
    "np",
    "comp0.075",
    "np*ln(1/mm)",
    "Rhat_0.1",
]


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a .parquet or .csv dataframe and ensure train_margin_median exists."""
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported data file extension: {p.suffix}")

    if "train_margin_median" not in df.columns:
        if "train_margin_dist" not in df.columns:
            raise ValueError(
                "Dataframe must contain train_margin_median or train_margin_dist."
            )
        df = df.copy()
        df["train_margin_median"] = df["train_margin_dist"].apply(
            lambda x: np.median(literal_eval(x))
        )
    return df


def build_measures(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build the complexity-measure arrays used by the K bootstrap."""
    med = df["train_margin_median"].to_numpy(float)
    nq = df["Number of Qubits"].to_numpy(float)
    npar = df["Number of Params"].to_numpy(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        measures = {
            "sqrt(nq)/mm": np.sqrt(nq) / med,
            "1/mm": 1.0 / med,
            "np/mm": npar / med,
            "np*sqrt(nq)": npar * np.sqrt(nq),
            "np": npar,
            "comp0.075": np.sqrt(nq) / med + 0.075 * np.sqrt(npar),
            "np*ln(1/mm)": npar * np.log(1.0 / med),
        }

    if "Rg_0.1" in df.columns:
        measures["Rhat_0.1"] = df["Rg_0.1"].to_numpy(float)
    return measures


def summarize_samples(samples: Sequence[float]) -> List[float]:
    """Return [2.5% quantile, median, 97.5% quantile]."""
    a = np.asarray(samples, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [float("nan"), float("nan"), float("nan")]
    return [
        float(np.quantile(a, 0.025)),
        float(np.median(a)),
        float(np.quantile(a, 0.975)),
    ]


def validate_names(measures: Mapping[str, np.ndarray], names: Sequence[str]) -> None:
    missing = [name for name in names if name not in measures]
    if missing:
        raise ValueError(
            "Missing requested measure(s): " + ", ".join(missing) + "\n"
            "For Rhat_0.1, the dataframe must contain an Rg_0.1 column."
        )


def bootstrap_k(
    df: pd.DataFrame,
    names: Sequence[str],
    *,
    B: int,
    rng: np.random.Generator,
    output_dir: str,
    checkpoint_every: int,
) -> Dict[str, object]:
    """Run the K(mu) seed-block bootstrap and write kboot JSON outputs."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    present_seeds = set(df["Random Seed"].unique())
    missing_seeds = [s for s in SEED_VALUES if s not in present_seeds]
    if missing_seeds:
        raise ValueError(f"Missing random-seed groups: {missing_seeds}")

    base_measures = build_measures(df)
    validate_names(base_measures, names)

    by_seed = {s: df[df["Random Seed"] == s] for s in SEED_VALUES}
    kboot: Dict[str, List[float]] = {name: [] for name in names}

    t0 = time.time()
    for b in range(B):
        draws = rng.integers(0, len(SEED_VALUES), size=len(SEED_VALUES))

        parts = []
        for new_label, draw in zip(SEED_VALUES, draws):
            part = by_seed[SEED_VALUES[int(draw)]].copy()
            part["Random Seed"] = new_label
            parts.append(part)

        df_b = pd.concat(parts, ignore_index=True)
        measures_b = build_measures(df_b)
        validate_names(measures_b, names)
        measures_b = {name: measures_b[name] for name in names}

        I_hats = E.run_ci(df_b, measures_b, names)
        k_scores = E.k_scores(I_hats, names)

        for name in names:
            kboot[name].append(float(k_scores[name][0]))

        if checkpoint_every > 0 and ((b + 1) % checkpoint_every == 0 or b == B - 1):
            partial = {
                "B_done": b + 1,
                "B_target": B,
                "K_ci": {
                    name: summarize_samples(kboot[name]) + [len(kboot[name])]
                    for name in names
                },
                "raw": kboot,
            }
            with (outdir / "kboot_partial.json").open("w") as f:
                json.dump(partial, f, indent=2)

            elapsed = time.time() - t0
            left = elapsed / (b + 1) * (B - b - 1)
            print(f"K boot {b + 1}/{B} ({elapsed:.0f}s, ~{left:.0f}s left)", flush=True)

    final = {
        "B": B,
        "K_ci": {name: summarize_samples(kboot[name]) for name in names},
        "raw": kboot,
    }
    with (outdir / "kboot.json").open("w") as f:
        json.dump(final, f, indent=2)

    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap K(mu) by resampling random seeds.")
    parser.add_argument(
        "--data",
        default="sprint_cache.parquet",
        help="Input dataframe, usually sprint_cache.parquet. Supports .parquet or .csv.",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=int(os.environ.get("B_K", "40")),
        help="Number of K-bootstrap replicates.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=20260405,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="bootstrap",
        help="Directory for kboot_partial.json and kboot.json.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Write kboot_partial.json every this many iterations. Use 0 to disable.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=DEFAULT_K_NAMES,
        help="Measure names to bootstrap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)
    df = load_dataframe(args.data)
    print(f"Loaded {len(df)} rows from {args.data}", flush=True)
    print("K-bootstrap measures:", ", ".join(args.names), flush=True)
    bootstrap_k(
        df,
        args.names,
        B=args.B,
        rng=rng,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Wrote {Path(args.output_dir) / 'kboot.json'}", flush=True)


if __name__ == "__main__":
    main()
