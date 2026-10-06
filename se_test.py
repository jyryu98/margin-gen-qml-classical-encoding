import itertools
import numpy as np
import os
from itertools import product
import json
import pandas as pd
from ast import literal_eval

HYPERPARAMETER_VALUES = {'Embedding': ['AngleEmbedding/RX', 'AngleEmbedding/RY', 'IQPEmbedding/1L', 'IQPEmbedding/2L'],
                         'Ansatz': ['BasicEntanglerLayers/RX', 'BasicEntanglerLayers/RY', 'StronglyEntanglingLayers/1L', 'SimplifiedTwoDesign/1L'],
                         'Number of ReUps': [4, 6, 8], 'Number of Qubits': [4, 6, 8], 'Batch Size': [32, 64, 128], 'Learning Rate': [0.001, 0.002, 0.005], 'Random Seed': [16, 13, 42, 65, 89],
                         'Dataset' : ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']}

def sign_error_with_dataset(results_df, complexity_measure, test_size, threshold):
    # Add dataset as a hyperparameter
    hyperparameter_values = HYPERPARAMETER_VALUES.copy()

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
                            if C_w1 != C_w2:
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

def hpwise_sign_error(results_df, hyperparameter_name, complexity_measure, test_size, threshold):
    # Add dataset as a hyperparameter
    hyperparameter_values = HYPERPARAMETER_VALUES.copy()

    se_list = []
    hyperparameter_to_change = hyperparameter_name
    hyperparameter_names = list(hyperparameter_values.keys())
    hyperparameter_names.remove('Random Seed')

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
                        if C_w1 != C_w2:
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

    results = pd.read_csv('./results_summary.csv')
    results['train_margin_median'] = results['train_margin_dist'].apply(lambda x: np.median(literal_eval(x)))

    complexity_measures = [lambda x: np.sqrt(x['Number of Qubits'])/x['train_margin_median'],
                           lambda x: 1/x['train_margin_median'],
                           lambda x: x['Number of Params']/x['train_margin_median'],
                           lambda x: x['Number of Params']*np.sqrt(x['Number of Qubits']),
                           lambda x: x['Number of Params']]

    se_lists = []
    for cmi, cm in enumerate(complexity_measures):
        se_list = sign_error_with_dataset(results, cm, 1000, 12)
        np.save('./se_test/cm_{}_se_list.npy'.format(cmi), np.array(se_list))
        se_lists.append(se_list)

    hyperparam_se_lists = {i: {} for i in range(5)}
    for cmi, cm in enumerate(complexity_measures):
        for hp in ['Embedding', 'Ansatz', 'Number of ReUps', 'Number of Qubits', 'Batch Size', 'Learning Rate', 'Dataset']:
            hyperparam_se_lists[cmi][hp] = hpwise_sign_error(results, hp, cm, test_size=1000, threshold=12)

    for cmi in range(5):
        hyperparam_se_lists[cmi]['common'] = []
        hyperparam_se_lists[cmi]['common'] = hyperparam_se_lists[cmi]['common'] + hyperparam_se_lists[cmi]['Ansatz']
        hyperparam_se_lists[cmi]['common'] = hyperparam_se_lists[cmi]['common'] + hyperparam_se_lists[cmi]['Number of ReUps']
        hyperparam_se_lists[cmi]['common'] = hyperparam_se_lists[cmi]['common'] + hyperparam_se_lists[cmi]['Number of Qubits']

    with open('./se_test/hyperparam_se_lists.json', 'w') as f:
        json.dump(hyperparam_se_lists, f)