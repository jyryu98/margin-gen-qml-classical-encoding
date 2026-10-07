import numpy as np
from itertools import product, combinations
import pandas as pd
from ast import literal_eval
import json

HYPERPARAMETER_VALUES = {'Embedding': ['AngleEmbedding/RX', 'AngleEmbedding/RY', 'IQPEmbedding/1L', 'IQPEmbedding/2L'],
                         'Ansatz': ['BasicEntanglerLayers/RX', 'BasicEntanglerLayers/RY', 'StronglyEntanglingLayers/1L', 'SimplifiedTwoDesign/1L'],
                         'Number of ReUps': [4, 6, 8], 'Number of Qubits': [4, 6, 8], 'Batch Size': [32, 64, 128], 'Learning Rate': [0.001, 0.002, 0.005], 'Random Seed': [16, 13, 42, 65, 89],
                         'Dataset': ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']}

def _count_both_strict_gt(g_a, mu_a, g_b, mu_b):
    """
    Count pairs (a,b) satisfying
        g_a > g_b and mu_a > mu_b
    in O((n_a+n_b) log n_b).
    """
    order_a = np.argsort(g_a, kind='stable')
    order_b = np.argsort(g_b, kind='stable')

    mu_levels = np.unique(mu_b)

    # Fenwick-tree indices for B points
    b_rank = np.searchsorted(mu_levels, mu_b, side='left')

    # Number of mu_levels strictly below each mu_a
    a_cut = np.searchsorted(mu_levels, mu_a, side='left')

    bit = np.zeros(len(mu_levels) + 1, dtype=np.int64)

    def add(idx):
        idx += 1
        while idx < len(bit):
            bit[idx] += 1
            idx += idx & -idx

    def prefix_sum(end):
        # Number of inserted elements having compressed index < end
        result = 0
        while end > 0:
            result += bit[end]
            end -= end & -end
        return result

    j = 0
    count = 0

    for ia in order_a:
        # Only insert B points with g_b < g_a
        while j < len(order_b) and g_b[order_b[j]] < g_a[ia]:
            ib = order_b[j]
            add(b_rank[ib])
            j += 1

        count += prefix_sum(a_cut[ia])

    return count


def joint_table_fast(g_a, mu_a, g_b, mu_b):
    """
    Rows:
        0: g_a >  g_b
        1: g_a <= g_b

    Columns:
        0: mu_a >  mu_b
        1: mu_a <= mu_b
    """
    n_a = len(g_a)
    n_b = len(g_b)
    total = n_a * n_b

    if total == 0:
        raise ValueError("Cannot construct a joint table from an empty group.")

    # Number satisfying g_a > g_b
    sorted_g_b = np.sort(g_b)
    G = int(np.searchsorted(
        sorted_g_b, g_a, side='left'
    ).sum())

    # Number satisfying mu_a > mu_b
    sorted_mu_b = np.sort(mu_b)
    M = int(np.searchsorted(
        sorted_mu_b, mu_a, side='left'
    ).sum())

    # Number satisfying both inequalities
    C = _count_both_strict_gt(
        g_a, mu_a, g_b, mu_b
    )

    counts = np.array([
        [C,     G - C],
        [M - C, total - G - M + C]
    ], dtype=np.int64)

    return counts

def mutual_information(table):
    p = table / table.sum()

    p_g = p.sum(axis=1, keepdims=True)
    p_mu = p.sum(axis=0, keepdims=True)

    denom = p_g * p_mu

    mask = p > 0
    return np.sum(
        p[mask] * np.log(p[mask] / denom[mask])
    )


def entropy_g(table):
    p = table / table.sum()
    p_g = p.sum(axis=1)

    mask = p_g > 0
    return -np.sum(p_g[mask] * np.log(p_g[mask]))

if __name__ == "__main__":
    total_results = pd.read_csv('./results_summary.csv')
    total_results['train_margin_median'] = total_results['train_margin_dist'].apply(lambda x: np.median(literal_eval(x)))

    # Calculate Complexity Measures
    margin = total_results['train_margin_median'].to_numpy(float)
    nqubits = total_results['Number of Qubits'].to_numpy(float)
    nparams = total_results['Number of Params'].to_numpy(float)

    total_results['mu_0'] = np.sqrt(nqubits) / margin
    total_results['mu_1'] = 1.0 / margin
    total_results['mu_2'] = nparams / margin
    total_results['mu_3'] = nparams * np.sqrt(nqubits)
    total_results['mu_4'] = nparams
    print('Finished preprocessing')

    metric_columns = ['mu_0', 'mu_1', 'mu_2', 'mu_3', 'mu_4']
    num_metrics = len(metric_columns)
    I_hats = {}

    # 1 Hyperparameter
    for S_component, values in HYPERPARAMETER_VALUES.items():
            print(S_component)
    
            groups = {}
    
            for value in values:
                subset = total_results[
                    total_results[S_component] == value
                ]
    
                if len(subset) == 0:
                    raise ValueError(
                        f"No rows for {S_component}={value}"
                    )
    
                groups[value] = {
                    'g': subset['Generalization Gap'].to_numpy(float),
                    'mu_0': subset['mu_0'].to_numpy(float),
                    'mu_1': subset['mu_1'].to_numpy(float),
                    'mu_2': subset['mu_2'].to_numpy(float),
                    'mu_3': subset['mu_3'].to_numpy(float),
                    'mu_4': subset['mu_4'].to_numpy(float),
                }
    
            I_sum = np.zeros(num_metrics)
            H_sum = np.zeros(num_metrics)
            n_tables = 0
    
            for a_value, b_value in product(values, repeat=2):
                A = groups[a_value]
                B = groups[b_value]
    
                for k, metric in enumerate(metric_columns):
                    table = joint_table_fast(
                        A['g'], A[metric],
                        B['g'], B[metric],
                    )
    
                    # If you decide that a model must not be
                    # compared with itself:
                    #
                    if a_value == b_value:
                        table[1, 1] -= len(A["g"])
    
                    I_sum[k] += mutual_information(table)
                    H_sum[k] += entropy_g(table)
    
                n_tables += 1
    
            I_hat_complexity = {}
    
            for k in range(num_metrics):
                I = I_sum[k] / n_tables
                H = H_sum[k] / n_tables
    
                if H <= 0:
                    I_hat = np.nan
                else:
                    I_hat = I / H
    
                I_hat_complexity[k] = float(I_hat)
            I_hats[S_component] = I_hat_complexity

    # 2 Hyperparameters
    for S_component1, S_component2 in combinations(
        HYPERPARAMETER_VALUES.keys(), 2
    ):
        print(S_component1, S_component2)

        values = list(product(
            HYPERPARAMETER_VALUES[S_component1],
            HYPERPARAMETER_VALUES[S_component2]
        ))

        groups = {}

        for value in values:
            subset = total_results[
                (total_results[S_component1] == value[0]) &
                (total_results[S_component2] == value[1])
            ]

            if len(subset) == 0:
                raise ValueError(
                    f"No rows for {(S_component1, S_component2)}={value}"
                )

            groups[value] = {
                'g': subset['Generalization Gap'].to_numpy(float),
                'mu_0': subset['mu_0'].to_numpy(float),
                'mu_1': subset['mu_1'].to_numpy(float),
                'mu_2': subset['mu_2'].to_numpy(float),
                'mu_3': subset['mu_3'].to_numpy(float),
                'mu_4': subset['mu_4'].to_numpy(float),
            }

        I_sum = np.zeros(num_metrics)
        H_sum = np.zeros(num_metrics)
        n_tables = 0

        for a_value, b_value in product(values, repeat=2):
            A = groups[a_value]
            B = groups[b_value]

            for k, metric in enumerate(metric_columns):
                table = joint_table_fast(
                    A['g'], A[metric],
                    B['g'], B[metric],
                )

                # Remove i == j self-comparisons
                if a_value == b_value:
                    table[1, 1] -= len(A["g"])

                I_sum[k] += mutual_information(table)
                H_sum[k] += entropy_g(table)

            n_tables += 1

        I_hat_complexity = {}

        for k in range(num_metrics):
            I = I_sum[k] / n_tables
            H = H_sum[k] / n_tables

            I_hat_complexity[k] = (
                float(I / H) if H > 0 else np.nan
            )

        I_hats[str((S_component1, S_component2))] = I_hat_complexity

        with open('./ci_test/cmi.json', 'w') as f:
            json.dump(I_hats, f)