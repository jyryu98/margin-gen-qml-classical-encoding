import pandas as pd 
import numpy as np 
from ast import literal_eval
import itertools
import json

HYPERPARAMETER_VALUES = {'Embedding': ['AngleEmbedding/RX', 'AngleEmbedding/RY', 'IQPEmbedding/1L', 'IQPEmbedding/2L'],
                         'Ansatz': ['BasicEntanglerLayers/RX', 'BasicEntanglerLayers/RY', 'StronglyEntanglingLayers/1L', 'SimplifiedTwoDesign/1L'],
                         'Number of ReUps': [4, 6, 8], 'Number of Qubits': [4, 6, 8], 'Batch Size': [32, 64, 128], 'Learning Rate': [0.001, 0.002, 0.005], 'Random Seed': [16, 13, 42, 65, 89],
                         'Dataset': ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']}

def sign_error_with_dataset(results_df, complexity_measure, test_size, threshold):
    # Add dataset as a hyperparameter
    hyperparameter_values = HYPERPARAMETER_VALUES.copy()
    hyperparameter_values['Dataset'] = ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']

    se_list = []
    hyperparameter_names = list(hyperparameter_values.keys())
    hyperparameter_names.remove('Random Seed')

    for hyperparameter_to_change in hyperparameter_names:
        # Constant hyperparameter names
        constant_hyperparameters = hyperparameter_names.copy()
        constant_hyperparameters.remove(hyperparameter_to_change)
        
        for constant_env in itertools.product(*[hyperparameter_values[i] for i in constant_hyperparameters]):
            # Constant environment
            env = {i:j for i,j in zip(constant_hyperparameters, constant_env)}
            for htc_idx1 in range(len(hyperparameter_values[hyperparameter_to_change])):
                for htc_idx2 in range(htc_idx1+1, len(hyperparameter_values[hyperparameter_to_change])):
                    # One different setting
                    htc_pair = [hyperparameter_values[hyperparameter_to_change][htc_idx1], hyperparameter_values[hyperparameter_to_change][htc_idx2]]
                    env1 = env.copy()
                    env1[hyperparameter_to_change] = htc_pair[0]
                    env2 = env.copy()
                    env2[hyperparameter_to_change] = htc_pair[1]

                    # Filter results
                    env1_filtered_results = results_df
                    for key in env1.keys():
                        env1_filtered_results = env1_filtered_results[env1_filtered_results[key] == env1[key]]
                    env2_filtered_results = results_df
                    for key in env2.keys():
                        env2_filtered_results = env2_filtered_results[env2_filtered_results[key] == env2[key]]

                    # Calculate sign error
                    se = 0
                    ks = []
                    for _, env1_result in env1_filtered_results.iterrows():
                        for _, env2_result in env2_filtered_results.iterrows():
                            G_w1 = env1_result['Generalization Gap']
                            G_w2 = env2_result['Generalization Gap']
                            C_w1 = complexity_measure(env1_result)
                            C_w2 = complexity_measure(env2_result)
                            if C_w1 != C_w2: # Cases where complexities are the same are removed due to a large fraction being present
                                epsilon = (G_w1-G_w2)/2
                                k = np.max([0, (1-2*np.exp(-2*test_size*epsilon**2))**2-0.5])
                                ks.append(k)
                                se += 0.5*k*(1-np.sign(G_w1-G_w2)*np.sign(C_w1-C_w2))
                    if np.sum(ks) != 0:
                        se = se / np.sum(ks)
                        neff = np.sum(ks)**2/np.sum(np.square(ks))
                        if neff >= threshold:
                            se_list.append(se)
    return se_list

def sign_error_with_dataset_without_ansatz_num_qubits(results_df, complexity_measure, test_size, threshold):
    # Add dataset as a hyperparameter
    hyperparameter_values = HYPERPARAMETER_VALUES.copy()
    hyperparameter_values['Dataset'] = ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']
    hyperparameter_values.pop('Ansatz')
    hyperparameter_values.pop('Number of Qubits')

    se_list = []
    hyperparameter_names = list(hyperparameter_values.keys())
    hyperparameter_names.remove('Random Seed')

    for hyperparameter_to_change in hyperparameter_names:
        # Constant hyperparameter names
        constant_hyperparameters = hyperparameter_names.copy()
        constant_hyperparameters.remove(hyperparameter_to_change)
        
        for constant_env in itertools.product(*[hyperparameter_values[i] for i in constant_hyperparameters]):
            # Constant environment
            env = {i:j for i,j in zip(constant_hyperparameters, constant_env)}
            for htc_idx1 in range(len(hyperparameter_values[hyperparameter_to_change])):
                for htc_idx2 in range(htc_idx1+1, len(hyperparameter_values[hyperparameter_to_change])):
                    # One different setting
                    htc_pair = [hyperparameter_values[hyperparameter_to_change][htc_idx1], hyperparameter_values[hyperparameter_to_change][htc_idx2]]
                    env1 = env.copy()
                    env1[hyperparameter_to_change] = htc_pair[0]
                    env2 = env.copy()
                    env2[hyperparameter_to_change] = htc_pair[1]

                    # Filter results
                    env1_filtered_results = results_df
                    for key in env1.keys():
                        env1_filtered_results = env1_filtered_results[env1_filtered_results[key] == env1[key]]
                    env2_filtered_results = results_df
                    for key in env2.keys():
                        env2_filtered_results = env2_filtered_results[env2_filtered_results[key] == env2[key]]

                    # Calculate sign error
                    se = 0
                    ks = []
                    for _, env1_result in env1_filtered_results.iterrows():
                        for _, env2_result in env2_filtered_results.iterrows():
                            G_w1 = env1_result['Generalization Gap']
                            G_w2 = env2_result['Generalization Gap']
                            C_w1 = complexity_measure(env1_result)
                            C_w2 = complexity_measure(env2_result)
                            if C_w1 != C_w2: # Cases where complexities are the same are removed due to a large fraction being present
                                epsilon = (G_w1-G_w2)/2
                                k = np.max([0, (1-2*np.exp(-2*test_size*epsilon**2))**2-0.5])
                                ks.append(k)
                                se += 0.5*k*(1-np.sign(G_w1-G_w2)*np.sign(C_w1-C_w2))
                    if np.sum(ks) != 0:
                        se = se / np.sum(ks)
                        neff = np.sum(ks)**2/np.sum(np.square(ks))
                        if neff >= threshold:
                            se_list.append(se)
    return se_list

if __name__ == '__main__':

    results = pd.read_csv('../results_summary.csv')
    results['train_margin_median'] = results['train_margin_dist'].apply(lambda x: np.median(literal_eval(x)))

    # Common c
    cs = np.linspace(0.01, 0.1, 19)
    se_lists = []
    for c in cs:
        cm = lambda x: np.sqrt(x['Number of Qubits'])/x['train_margin_median'] + c * np.sqrt(x['Number of Params'])
        se_list = sign_error_with_dataset(results, cm, 1000, 12)
        np.save('./se_test/composite_measure_{}c_se_list.npy'.format('{:.2e}'.format(c)), np.array(se_list))
        se_lists.append(se_list)

    # (Ansatz, Num Qubits) c
    c_gridsearch_results = {}
    for num_qubits in HYPERPARAMETER_VALUES['Number of Qubits']:
        for ansatz_type in HYPERPARAMETER_VALUES['Ansatz']:
            print(ansatz_type, num_qubits)
            c_gridsearch_results['{}_{}Q'.format(ansatz_type, num_qubits)] = {}
            cs = np.linspace(0, 0.2, 41)
            subresults = results[results['Ansatz'] == ansatz_type]
            subresults = subresults[subresults['Number of Qubits'] == num_qubits]
            for c in cs:
                cm = lambda x: np.sqrt(x['Number of Qubits'])/x['train_margin_median'] + c * np.sqrt(x['Number of Params'])
                se_list = sign_error_with_dataset_without_ansatz_num_qubits(subresults, cm, 1000, 12)
                c_gridsearch_results['{}_{}Q'.format(ansatz_type, num_qubits)][c] = se_list

    with open('./se_test/c_gridsearch_results.json', 'w') as f:
        json.dump(c_gridsearch_results, f)

    c_lookup_table = {}
    for data_idx, setting_name in enumerate(c_gridsearch_results.keys()):
        c_lookup_table[setting_name] = cs[np.argmin([np.percentile(se_list, 90) for se_list in [c_gridsearch_results[setting_name][c] for c in cs]])]

    with open('./se_test/c_lookup_table.json', 'w') as f:
        json.dump(c_lookup_table, f)