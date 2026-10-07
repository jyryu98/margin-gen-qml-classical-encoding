"""
extra_measures_test.py
======================

Computes the conditional-independence (CI) test score K(mu) and the
sign-error (SE) statistics for two ADDITIONAL complexity measures, using
exactly the paper's criteria (ci_test.py / se_test.py):

    mu_a = n_p * ln(1 / m_m)
    mu_b = n_p * ln(n_p / m_m)

Each is evaluated twice:
    (median) m_m = np.median(literal_eval(train_margin_dist))   <- paper's m_m
    (mean)   m_m = the 'train_margin_mean' column               <- pre-zeroing

Design / fidelity notes
-----------------------
* CI reuses ci_test.joint_table_fast / mutual_information / entropy_g over the
  SAME 36 conditioning sets (8 single hyperparameters + 28 pairs), accumulated
  with the identical ordered-pair loop and the a==b self-comparison correction.
* SE is a vectorized port of se_test.hpwise_sign_error (test_size=1000,
  threshold n_eff >= 12), emitting environment SE values in the EXACT same
  (constant-env major, pair minor) order as se_test, so results compare
  element-by-element to the saved artifacts.
* m_m (median) is rebuilt from results_summary.csv via
  np.median(literal_eval(train_margin_dist)) -- bit-exact to what the original
  scripts used -- cached to parquet so reruns are fast.

Degenerate cases (reported, never silently dropped):
* median m_m == 0  -> 1/m_m = +inf, ln(+inf) = +inf.  This is the SAME +inf
  that the paper's 1/m_m measure already produces, and the strict-inequality
  comparisons in CI/SE handle +inf consistently (inf > finite; inf !> inf).
  Such runs are KEPT for the median-based variants.
* mean   m_m <  0  -> ln of a negative number = NaN. The measure is UNDEFINED
  on those runs, so they are EXCLUDED from that variant's CI/SE (and the count
  is reported). mean m_m == 0 -> +inf, kept like the median==0 case.

--validate also recomputes the 5 original measures and checks them against
ci_test/cmi.json and se_test/cm_*_se_list.npy + hyperparam_se_lists.json.

Outputs (never overwrites existing results):
    ci_test/cmi_extra.json
    se_test/hyperparam_se_lists_extra.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from ast import literal_eval
from itertools import product, combinations

import numpy as np
import pandas as pd

import ci_test  # reuse joint_table_fast / mutual_information / entropy_g

HYPERPARAMETER_VALUES = {
    'Embedding': ['AngleEmbedding/RX', 'AngleEmbedding/RY', 'IQPEmbedding/1L', 'IQPEmbedding/2L'],
    'Ansatz': ['BasicEntanglerLayers/RX', 'BasicEntanglerLayers/RY', 'StronglyEntanglingLayers/1L', 'SimplifiedTwoDesign/1L'],
    'Number of ReUps': [4, 6, 8],
    'Number of Qubits': [4, 6, 8],
    'Batch Size': [32, 64, 128],
    'Learning Rate': [0.001, 0.002, 0.005],
    'Random Seed': [16, 13, 42, 65, 89],
    'Dataset': ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6'],
}

# Order of hyperparameters used by se_test (dict order minus 'Random Seed').
SE_HP_ORDER = ['Embedding', 'Ansatz', 'Number of ReUps', 'Number of Qubits',
               'Batch Size', 'Learning Rate', 'Dataset']
COMMON_HPS = ['Ansatz', 'Number of ReUps', 'Number of Qubits']

CSV_PATH = './results_summary.csv'
CACHE_PATH = os.environ.get('EXTRA_CACHE', './extra_measures_cache.parquet')

# c for the paper's composite sqrt(nq)/mm + c*sqrt(np); 0.075 is the argmin of
# the global 90th-percentile sign-error over the paper's grid (reproduces 0.7118).
COMPOSITE_C = 0.075


# --------------------------------------------------------------------------- #
# Data loading: rebuild median margin from the full CSV (chunked, bit-exact).  #
# --------------------------------------------------------------------------- #
def load_data():
    if os.path.exists(CACHE_PATH):
        print(f'Loading cached data from {CACHE_PATH}')
        return pd.read_parquet(CACHE_PATH)

    print(f'Building median-margin cache from {CSV_PATH} (chunked)...')
    keep = ['Dataset', 'Embedding', 'Ansatz', 'Number of ReUps', 'Number of Qubits',
            'Batch Size', 'Learning Rate', 'Random Seed', 'Number of Params',
            'Generalization Gap', 'train_margin_mean']
    parts = []
    t0 = time.time()
    for ch in pd.read_csv(CSV_PATH, chunksize=1000, usecols=keep + ['train_margin_dist']):
        med = ch['train_margin_dist'].apply(lambda x: np.median(literal_eval(x))).to_numpy()
        sub = ch[keep].copy()
        sub['train_margin_median'] = med
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(CACHE_PATH)
    print(f'Cache built: {len(df)} rows in {time.time() - t0:.0f}s -> {CACHE_PATH}')
    return df


# --------------------------------------------------------------------------- #
# Complexity measures.                                                         #
# --------------------------------------------------------------------------- #
def build_measures(df):
    """Return (ordered dict name->array, meta dict name->degeneracy counts)."""
    med = df['train_margin_median'].to_numpy(float)
    mean = df['train_margin_mean'].to_numpy(float)
    nq = df['Number of Qubits'].to_numpy(float)
    npar = df['Number of Params'].to_numpy(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        measures = {
            # ---- 5 originals (indices 0..4 map to cmi.json '0'..'4') ----
            'sqrt(nq)/mm':      np.sqrt(nq) / med,
            '1/mm':             1.0 / med,
            'np/mm':            npar / med,
            'np*sqrt(nq)':      npar * np.sqrt(nq),
            'np':               npar,
            # ---- composite from the paper ----
            f'sqrt(nq)/mm+{COMPOSITE_C}sqrt(np)': np.sqrt(nq) / med + COMPOSITE_C * np.sqrt(npar),
            # ---- 4 new variants ----
            'np*ln(1/mm)[med]':  npar * np.log(1.0 / med),
            'np*ln(np/mm)[med]': npar * np.log(npar / med),
            'np*ln(1/mm)[mean]': npar * np.log(1.0 / mean),
            'np*ln(np/mm)[mean]': npar * np.log(npar / mean),
        }

    meta = {}
    for name, arr in measures.items():
        meta[name] = {
            'n_nan': int(np.isnan(arr).sum()),
            'n_posinf': int(np.isposinf(arr).sum()),
            'n_neginf': int(np.isneginf(arr).sum()),
        }
    # degeneracy sources (reported)
    meta['_median_zero'] = int((med == 0).sum())
    meta['_mean_zero'] = int((mean == 0).sum())
    meta['_mean_neg'] = int((mean < 0).sum())
    return measures, meta


ORIGINAL_NAMES = ['sqrt(nq)/mm', '1/mm', 'np/mm', 'np*sqrt(nq)', 'np']
NEW_NAMES = ['np*ln(1/mm)[med]', 'np*ln(np/mm)[med]',
             'np*ln(1/mm)[mean]', 'np*ln(np/mm)[mean]']


# --------------------------------------------------------------------------- #
# CI test: K(mu) = min over 36 conditioning sets of I_hat = I / H.            #
# --------------------------------------------------------------------------- #
def _ihat_for_group_set(groups, names):
    """Replicate ci_test's accumulation over all ordered (a,b) group pairs.

    `groups[value]` holds 'g' (gap) plus one array per measure name. For
    measures containing NaN, the NaN rows (and their gaps) are dropped per
    measure -- the measure is only scored where it is defined.
    """
    values = list(groups.keys())
    I_sum = {n: 0.0 for n in names}
    H_sum = {n: 0.0 for n in names}
    n_tables = {n: 0 for n in names}  # per-measure: a group may be empty only after NaN-drop

    # Pre-filter per (value, measure): finite-mask drops NaN (keeps +/-inf).
    filt = {}
    for v in values:
        g = groups[v]['g']
        filt[v] = {}
        for n in names:
            mu = groups[v][n]
            if np.isnan(mu).any():
                m = ~np.isnan(mu)
                filt[v][n] = (g[m], mu[m])
            else:
                filt[v][n] = (g, mu)

    for a, b in product(values, repeat=2):
        for n in names:
            g_a, mu_a = filt[a][n]
            g_b, mu_b = filt[b][n]
            if len(g_a) == 0 or len(g_b) == 0:
                continue
            table = ci_test.joint_table_fast(g_a, mu_a, g_b, mu_b)
            if a == b:
                table[1, 1] -= len(g_a)
            I_sum[n] += ci_test.mutual_information(table)
            H_sum[n] += ci_test.entropy_g(table)
            n_tables[n] += 1

    out = {}
    for n in names:
        if n_tables[n] == 0:
            out[n] = float('nan')
            continue
        I = I_sum[n] / n_tables[n]
        H = H_sum[n] / n_tables[n]
        out[n] = float(I / H) if H > 0 else float('nan')
    return out


def run_ci(df, measures, names):
    """Return I_hats[cond_set_str][name] over the 36 conditioning sets."""
    gap = df['Generalization Gap'].to_numpy(float)
    I_hats = {}

    # 1 hyperparameter (8 sets, including 'Random Seed').
    for S, vals in HYPERPARAMETER_VALUES.items():
        groups = {}
        for v in vals:
            idx = (df[S] == v).to_numpy()
            if idx.sum() == 0:
                raise ValueError(f'No rows for {S}={v}')
            groups[v] = {'g': gap[idx], **{n: measures[n][idx] for n in names}}
        I_hats[S] = _ihat_for_group_set(groups, names)
        print(f'  CI done: {S}')

    # 2 hyperparameters (28 sets).
    for S1, S2 in combinations(HYPERPARAMETER_VALUES.keys(), 2):
        vals = list(product(HYPERPARAMETER_VALUES[S1], HYPERPARAMETER_VALUES[S2]))
        groups = {}
        for v in vals:
            idx = ((df[S1] == v[0]) & (df[S2] == v[1])).to_numpy()
            if idx.sum() == 0:
                raise ValueError(f'No rows for {(S1, S2)}={v}')
            groups[v] = {'g': gap[idx], **{n: measures[n][idx] for n in names}}
        I_hats[str((S1, S2))] = _ihat_for_group_set(groups, names)
        print(f'  CI done: {S1} , {S2}')

    return I_hats


def k_scores(I_hats, names):
    """K(mu) = min over conditioning sets; also return the minimizing set."""
    out = {}
    for n in names:
        best_key, best_val = None, np.inf
        for key, d in I_hats.items():
            v = d[n]
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and v < best_val:
                best_val, best_key = v, key
        out[n] = (best_val, best_key)
    return out


# --------------------------------------------------------------------------- #
# SE test: vectorized port of se_test.hpwise_sign_error.                      #
# --------------------------------------------------------------------------- #
def _encode(df):
    """Integer code (index within value list) for every hyperparameter, +seed slot."""
    codes = {}
    for hp, vals in HYPERPARAMETER_VALUES.items():
        mapping = {v: i for i, v in enumerate(vals)}
        codes[hp] = df[hp].map(mapping).to_numpy()
    return codes


def se_hpwise(gap, cvals, codes, hp, test_size=1000, threshold=12):
    """SE list for changing hyperparameter `hp`, in se_test's exact emit order.

    The 7-non-seed-hyperparameter grid is complete (5 seeds per cell), so each
    environment is a (constant-env, value-pair) with a 5x5 seed cross product.
    NaN measure values are masked out of comparisons (measure undefined there).
    """
    constant = [h for h in SE_HP_ORDER if h != hp]
    const_sizes = [len(HYPERPARAMETER_VALUES[h]) for h in constant]
    m = len(HYPERPARAMETER_VALUES[hp])
    n_base = int(np.prod(const_sizes))
    n_seed = len(HYPERPARAMETER_VALUES['Random Seed'])

    # base_code folds constant hps in SE order (first slowest) == itertools.product order.
    base_code = np.zeros(len(gap), dtype=np.int64)
    for h, size in zip(constant, const_sizes):
        base_code = base_code * size + codes[h]
    h_idx = codes[hp]
    seed_slot = codes['Random Seed']

    G = np.zeros((n_base, m, n_seed))
    C = np.zeros((n_base, m, n_seed))
    V = np.zeros((n_base, m, n_seed), dtype=bool)  # True where measure is defined (finite or +/-inf)
    G[base_code, h_idx, seed_slot] = gap
    C[base_code, h_idx, seed_slot] = cvals
    V[base_code, h_idx, seed_slot] = ~np.isnan(cvals)

    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    n_pairs = len(pairs)

    se_arr = np.full((n_base, n_pairs), np.nan)
    keep_arr = np.zeros((n_base, n_pairs), dtype=bool)

    for pidx, (i, j) in enumerate(pairs):
        G1 = G[:, i, :][:, :, None]          # (n_base, 5, 1)
        G2 = G[:, j, :][:, None, :]          # (n_base, 1, 5)
        C1 = C[:, i, :][:, :, None]
        C2 = C[:, j, :][:, None, :]
        v1 = V[:, i, :][:, :, None]
        v2 = V[:, j, :][:, None, :]

        dG = G1 - G2
        eps = dG / 2.0
        k = np.maximum(0.0, (1.0 - 2.0 * np.exp(-2.0 * test_size * eps ** 2)) ** 2 - 0.5)
        sgnG = np.sign(dG)
        neq = (v1 & v2) & (C1 != C2)
        sgnC = np.sign(C1 - C2)
        term = 0.5 * k * (1.0 - sgnG * sgnC)

        kk = np.where(neq, k, 0.0)
        tt = np.where(neq, term, 0.0)
        sumk = kk.sum(axis=(1, 2))
        sumk2 = (kk * kk).sum(axis=(1, 2))
        senum = tt.sum(axis=(1, 2))

        ok = sumk != 0
        safe = np.where(ok, sumk, 1.0)
        se = np.where(ok, senum / safe, np.nan)
        neff = np.where(ok, sumk ** 2 / np.where(ok, sumk2, 1.0), 0.0)
        keep = ok & (neff >= threshold)

        se_arr[:, pidx] = se
        keep_arr[:, pidx] = keep

    # Emit in (constant-env major, pair minor) order == se_test ordering.
    out = []
    for base in range(n_base):
        row_keep = keep_arr[base]
        if row_keep.any():
            row_se = se_arr[base]
            for pidx in range(n_pairs):
                if row_keep[pidx]:
                    out.append(float(row_se[pidx]))
    return out


def run_se(df, measures, names):
    """Return se_lists[name] = {hp: [...], 'common': [...], 'All': [...]}."""
    gap = df['Generalization Gap'].to_numpy(float)
    codes = _encode(df)
    out = {}
    for n in names:
        cvals = measures[n]
        per_hp = {}
        for hp in SE_HP_ORDER:
            per_hp[hp] = se_hpwise(gap, cvals, codes, hp)
        per_hp['common'] = sum((per_hp[h] for h in COMMON_HPS), [])
        per_hp['All'] = sum((per_hp[h] for h in SE_HP_ORDER), [])
        out[n] = per_hp
        print(f'  SE done: {n}  (All n={len(per_hp["All"])}, common n={len(per_hp["common"])})')
    return out


def se_stats(lst):
    a = np.asarray(lst, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (float('nan'), float('nan'), float('nan'), 0)
    return (float(np.quantile(a, 0.90)), float(np.mean(a)), float(np.max(a)), int(len(a)))


def report(ks, se_lists, meta, names):
    print('\n' + '=' * 110)
    print('DEGENERACY COUNTS (reported, not dropped silently)')
    print('=' * 110)
    print(f"  median m_m == 0  (-> +inf for 1/mm, np/mm, sqrt(nq)/mm, and median-based ln measures): {meta['_median_zero']}")
    print(f"  mean   m_m == 0  (-> +inf for mean-based ln measures):                                  {meta['_mean_zero']}")
    print(f"  mean   m_m <  0  (-> NaN, EXCLUDED from mean-based ln measures' CI/SE):                 {meta['_mean_neg']}")
    for n in NEW_NAMES:
        print(f"    {n:22s}: NaN={meta[n]['n_nan']:5d}  +inf={meta[n]['n_posinf']:4d}  -inf={meta[n]['n_neginf']:4d}")

    print('\n' + '=' * 110)
    print('MAIN TABLE: K(mu) and sign-error (All / Common)')
    print('=' * 110)
    hdr = (f"{'measure':24s} {'K(mu)':>11s}  {'min-cond-set':28s}  "
           f"{'All q90':>8s} {'All mean':>8s} {'All max':>7s} {'All n':>6s}  "
           f"{'Com q90':>8s} {'Com mean':>8s} {'Com max':>7s} {'Com n':>6s}")
    print(hdr)
    print('-' * len(hdr))

    def line(n, label=None):
        kval, kset = ks[n]
        aq, am, ax, an = se_stats(se_lists[n]['All'])
        cq, cm, cx, cn = se_stats(se_lists[n]['common'])
        kset_s = str(kset)
        if len(kset_s) > 28:
            kset_s = kset_s[:27] + '~'
        print(f"{(label or n):24s} {kval:11.4e}  {kset_s:28s}  "
              f"{aq:8.4f} {am:8.4f} {ax:7.4f} {an:6d}  "
              f"{cq:8.4f} {cm:8.4f} {cx:7.4f} {cn:6d}")

    print('-- 5 originals --')
    for n in ORIGINAL_NAMES:
        line(n)
    print('-- composite (c=%.3f) --' % COMPOSITE_C)
    comp = [n for n in names if n.startswith('sqrt(nq)/mm+')][0]
    line(comp, label='sqrt(nq)/mm+c*sqrt(np)')
    print('-- 4 new variants --')
    for n in NEW_NAMES:
        line(n)

    # Per-hyperparameter q90 breakdown (Figure-3 style) for 1/mm and the new measures.
    print('\n' + '=' * 110)
    print('PER-HYPERPARAMETER 90th-PERCENTILE SIGN-ERROR (Figure 3 style)')
    print('=' * 110)
    focus = ['1/mm', 'np/mm', 'np'] + NEW_NAMES
    hdr2 = f"{'measure':24s} " + ' '.join(f"{hp[:9]:>10s}" for hp in SE_HP_ORDER)
    print(hdr2)
    print('-' * len(hdr2))
    for n in focus:
        cells = []
        for hp in SE_HP_ORDER:
            q = se_stats(se_lists[n][hp])[0]
            cells.append(f"{q:10.4f}")
        print(f"{n:24s} " + ' '.join(cells))


def main():
    warnings.filterwarnings('ignore')
    t0 = time.time()

    df = load_data()
    assert len(df) == 19440, f'expected 19440 rows, got {len(df)}'
    measures, meta = build_measures(df)

    names = ORIGINAL_NAMES + [n for n in measures if n.startswith('sqrt(nq)/mm+')] + NEW_NAMES

    print('\nRunning CI test (36 conditioning sets)...')
    I_hats = run_ci(df, measures, names)
    ks = k_scores(I_hats, names)

    print('\nRunning SE test (vectorized)...')
    se_lists = run_se(df, measures, names)

    report(ks, se_lists, meta, names)

    # ---- write outputs (new measures only), without overwriting existing ----
    os.makedirs('./ci_test', exist_ok=True)
    os.makedirs('./se_test', exist_ok=True)

    cmi_extra = {}
    for key, d in I_hats.items():
        cmi_extra[key] = {n: d[n] for n in NEW_NAMES}
    with open('./ci_test/cmi_extra.json', 'w') as f:
        json.dump(cmi_extra, f, indent=2)

    se_extra = {}
    for n in NEW_NAMES:
        se_extra[n] = {hp: se_lists[n][hp] for hp in SE_HP_ORDER}
        se_extra[n]['common'] = se_lists[n]['common']
        se_extra[n]['All'] = se_lists[n]['All']
    with open('./se_test/hyperparam_se_lists_extra.json', 'w') as f:
        json.dump(se_extra, f)

    print(f'\nWrote ci_test/cmi_extra.json and se_test/hyperparam_se_lists_extra.json')
    print(f'Total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
