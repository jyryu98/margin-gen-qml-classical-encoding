import datetime
import os
import random
import json
import torch
import numpy as np
from experiment import build_model_from_hyperparameter_dict, train

HYPERPARAMETER_VALUES = {'Embedding': ['AngleEmbedding/RX', 'AngleEmbedding/RY', 'IQPEmbedding/1L', 'IQPEmbedding/2L'],
                         'Ansatz': ['BasicEntanglerLayers/RX', 'BasicEntanglerLayers/RY', 'StronglyEntanglingLayers/1L', 'SimplifiedTwoDesign/1L'],
                         'Number of ReUps': [4, 6, 8], 'Number of Qubits': [4, 6, 8], 'Batch Size': [32, 64, 128], 'Learning Rate': [0.001, 0.002, 0.005], 'Random Seed': [16, 13, 42, 65, 89]}

if __name__ == '__main__':

    for dataset in ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']:

        train_x = torch.load('./{}/train_x.pt'.format(dataset), weights_only = False)
        train_y = torch.load('./{}/train_y.pt'.format(dataset), weights_only = False)
        test_x = torch.load('./{}/test_x.pt'.format(dataset), weights_only = False)
        test_y = torch.load('./{}/test_y.pt'.format(dataset), weights_only = False)

        for embedding in HYPERPARAMETER_VALUES['Embedding']:
            for ansatz in HYPERPARAMETER_VALUES['Ansatz']:
                for num_reups in HYPERPARAMETER_VALUES['Number of ReUps']:
                    for num_qubits in HYPERPARAMETER_VALUES['Number of Qubits']:
                        for batchsize in HYPERPARAMETER_VALUES['Batch Size']:
                            for lr in HYPERPARAMETER_VALUES['Learning Rate']:
                                for seed in HYPERPARAMETER_VALUES['Random Seed']:
                                
                                    # Set random seeds
                                    torch.manual_seed(seed)
                                    random.seed(seed)
                                    np.random.seed(seed)
                                    
                                    # Make save folder
                                    now = datetime.datetime.now()
                                    date_string = now.strftime("%Y%m%d_%H%M%S")
                                    save_path = os.path.join(dataset, date_string)
                                    os.mkdir(save_path)
                                    
                                    # Save hyperparameters
                                    hyperparameter_dict = {'Embedding': embedding, 'Ansatz': ansatz, 'Number of ReUps': num_reups,
                                                        'Number of Qubits': num_qubits, 'Batch Size': batchsize, 'Learning Rate': lr, 'Random Seed': seed}
                                    with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
                                        json.dump(hyperparameter_dict, f)
                                    print(hyperparameter_dict)
                                    
                                    # Initialize model
                                    model = build_model_from_hyperparameter_dict(hyperparameter_dict)

                                    # Initialize data
                                    train_x_stacked = np.concatenate([train_x]*int(hyperparameter_dict['Number of Qubits']/2), axis = 1)
                                    test_x_stacked = np.concatenate([test_x]*int(hyperparameter_dict['Number of Qubits']/2), axis = 1)

                                    # Train
                                    train(train_x_stacked, train_y, lr, batchsize, model, save_path, seed, convergence_interval=500)