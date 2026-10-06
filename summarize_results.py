import pandas as pd
import json
import os
import torch

HYPERPARAMETER_VALUES = {'Embedding': ['AngleEmbedding/RX', 'AngleEmbedding/RY', 'IQPEmbedding/1L', 'IQPEmbedding/2L'],
                         'Ansatz': ['BasicEntanglerLayers/RX', 'BasicEntanglerLayers/RY', 'StronglyEntanglingLayers/1L', 'SimplifiedTwoDesign/1L'],
                         'Number of ReUps': [4, 6, 8], 'Number of Qubits': [4, 6, 8], 'Batch Size': [32, 64, 128], 'Learning Rate': [0.001, 0.002, 0.005], 'Random Seed': [16, 13, 42, 65, 89]}

if __name__ == '__main__':

    total_results  = pd.DataFrame(columns = ['Dataset', 'Embedding', 'Ansatz', 'Number of ReUps', 'Number of Qubits', 'Batch Size', 'Learning Rate', 'Random Seed', 
                                        'train_acc', 'train_margin_dist', 'train_margin_boxplot', 'train_margin_mean', 'test_acc', 'Number of Params', 'Generalization Gap'])

    for dataset in ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']:

        search_space = [os.path.join(dataset, fn) for fn in os.listdir('./{}/'.format(dataset)) if '2026' in fn]

        for embedding in HYPERPARAMETER_VALUES['Embedding']:
            for ansatz in HYPERPARAMETER_VALUES['Ansatz']:
                for num_reups in HYPERPARAMETER_VALUES['Number of ReUps']:
                    for num_qubits in HYPERPARAMETER_VALUES['Number of Qubits']:
                        for batchsize in HYPERPARAMETER_VALUES['Batch Size']:
                            for lr in HYPERPARAMETER_VALUES['Learning Rate']:
                                for seed in HYPERPARAMETER_VALUES['Random Seed']:
                                    hyperparameter_dict = {'Embedding': embedding, 'Ansatz': ansatz, 'Number of ReUps': num_reups,
                                                                                        'Number of Qubits': num_qubits, 'Batch Size': batchsize, 'Learning Rate': lr, 'Random Seed': seed}
                                    found = False
                                    for dir in search_space:
                                        if not found:
                                            with open(os.path.join(dir, 'hyperparameters.json'), 'r') as f:
                                                run_hyperparameters = json.load(f)
                                            if run_hyperparameters == hyperparameter_dict:
                                                found = True
                                                search_space.remove(dir)
                                                found_dir = dir

                                    if not found:
                                        print('HYPERPARAMETER NOT FOUND: {}'.format(hyperparameter_dict))

                                    if found:
                                        with open(os.path.join(found_dir, 'score_margin.json'), 'r') as f:
                                            score_margin_dict = json.load(f)
                                        row_to_write = hyperparameter_dict|score_margin_dict
                                        row_to_write['Dataset'] = dataset
                                        state_dict = torch.load(os.path.join(found_dir, 'model_params.pth'))
                                        row_to_write['Number of Params'] = sum(p.numel() for p in state_dict.values())
                                        row_to_write['Generalization Gap'] = row_to_write['train_acc'] - row_to_write['test_acc']
                                        total_results.loc[len(total_results)] = row_to_write

    total_results.to_csv('results_summary.csv', index = False)