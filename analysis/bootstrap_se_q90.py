"""
bootstrap_se_q90.py

This computes SE lists using extra_measures_test.run_se and then
bootstraps the 90th percentile of each SE list.
"""

from __future__ import annotations

import argparse
import json
import os
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import extra_measures_test as E


ORIG = [
    "sqrt(nq)/mm",
    "1/mm",
    "np/mm",
    "np*sqrt(nq)",
    "np",
]
COMP = "sqrt(nq)/mm+0.075sqrt(np)"
DEFAULT_SE_NAMES = ORIG + [COMP, "np*ln(1/mm)", "np*ln(np/mm)"]


def load_dataframe(path: str | None) -> pd.DataFrame:
    """Load data. If path is None, use extra_measures_test.load_data()."""
    if path is None:
        return E.load_data()

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
    """Build the measures needed for the SE q90 bootstrap."""
    med = df["train_margin_median"].to_numpy(float)
    nq = df["Number of Qubits"].to_numpy(float)
    npar = df["Number of Params"].to_numpy(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "sqrt(nq)/mm": np.sqrt(nq) / med,
            "1/mm": 1.0 / med,
            "np/mm": npar / med,
            "np*sqrt(nq)": npar * np.sqrt(nq),
            "np": npar,
            COMP: np.sqrt(nq) / med + 0.075 * np.sqrt(npar),
            "np*ln(1/mm)": npar * np.log(1.0 / med),
            "np*ln(np/mm)": npar * np.log(npar / med),
        }


def se_stats(values: Sequence[float]) -> List[float | int]:
    """Return [q90, mean, max, n] for finite SE values."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [float("nan"), float("nan"), float("nan"), 0]
    return [
        float(np.quantile(a, 0.90)),
        float(np.mean(a)),
        float(np.max(a)),
        int(len(a)),
    ]


def bootstrap_q90(values: Sequence[float], *, B: int, rng: np.random.Generator) -> List[float]:
    """Return [2.5%, 50%, 97.5%] bootstrap CI for the SE 90th percentile."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [float("nan"), float("nan"), float("nan")]

    idx = rng.integers(0, len(a), size=(B, len(a)))
    q_samples = np.quantile(a[idx], 0.90, axis=1)
    return [
        float(np.quantile(q_samples, 0.025)),
        float(np.quantile(q_samples, 0.5)),
        float(np.quantile(q_samples, 0.975)),
    ]


def bootstrap_se_q90(
    df: pd.DataFrame,
    names: Sequence[str],
    *,
    B: int,
    rng: np.random.Generator,
    output_dir: str,
) -> Dict[str, object]:
    """Compute SE lists, bootstrap their q90s, and write JSON output."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    measures = build_measures(df)
    missing = [name for name in names if name not in measures]
    if missing:
        raise ValueError("Missing requested measure(s): " + ", ".join(missing))
    measures = {name: measures[name] for name in names}

    se_lists = E.run_se(df, measures, names)

    se_summary: Dict[str, Dict[str, List[float | int]]] = {}
    se_q90_ci: Dict[str, Dict[str, List[float]]] = {}

    # This order matters for exact reproducibility with a fixed RNG seed.
    for name in names:
        se_summary[name] = {
            "All": se_stats(se_lists[name]["All"]),
            "Common": se_stats(se_lists[name]["common"]),
        }
        se_q90_ci[name] = {
            "All": bootstrap_q90(se_lists[name]["All"], B=B, rng=rng),
            "Common": bootstrap_q90(se_lists[name]["common"], B=B, rng=rng),
        }

    payload = {
        "B": B,
        "SE": se_summary,
        "SE_q90_ci": se_q90_ci,
    }
    with (outdir / "se_q90_bootstrap.json").open("w") as f:
        json.dump(payload, f, indent=2)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap SE q90 uncertainty.")
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Optional dataframe path. Supports .parquet or .csv. "
            "If omitted, extra_measures_test.load_data() is used."
        ),
    )
    parser.add_argument(
        "--B",
        type=int,
        default=int(os.environ.get("B_SE", "2000")),
        help="Number of bootstrap replicates for each SE list.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=20260401,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="bootstrap",
        help="Directory for se_q90_bootstrap.json.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=DEFAULT_SE_NAMES,
        help="Measure names to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)
    df = load_dataframe(args.data)
    print(f"Loaded {len(df)} rows.", flush=True)
    print("SE-bootstrap measures:", ", ".join(args.names), flush=True)
    bootstrap_se_q90(
        df,
        args.names,
        B=args.B,
        rng=rng,
        output_dir=args.output_dir,
    )
    print(f"Wrote {Path(args.output_dir) / 'se_q90_bootstrap.json'}", flush=True)


if __name__ == "__main__":
    main()
