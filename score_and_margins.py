import datetime
import os
import random
import json
import torch
import numpy as np
from experiment import build_model_from_hyperparameter_dict, score_and_margins

if __name__ == '__main__':

    for dataset in ['Fashion4v9', 'KMNIST0v5', 'MNIST1v6']:
        # Load dataset
        train_x = torch.load('./{}/train_x.pt'.format(dataset), weights_only = False)
        train_y = torch.load('./{}/train_y.pt'.format(dataset), weights_only = False)
        test_x = torch.load('./{}/test_x.pt'.format(dataset), weights_only = False)
        test_y = torch.load('./{}/test_y.pt'.format(dataset), weights_only = False)

        for folder_name in os.listdir('./{}'.format(dataset)):

            # Make model and load params
            if '2026' in folder_name:
                with open(os.path.join(dataset, folder_name, 'hyperparameters.json'), 'r') as f:
                    hyperparameter_dict = json.load(f)
                model_parameters = torch.load(os.path.join(dataset, folder_name, 'model_params.pth'))
                model = build_model_from_hyperparameter_dict(hyperparameter_dict)
                model.load_state_dict(model_parameters)

                # Initialize data
                train_x_stacked = torch.tensor(np.concatenate([train_x]*int(hyperparameter_dict['Number of Qubits']/2), axis = 1))
                test_x_stacked = torch.tensor(np.concatenate([test_x]*int(hyperparameter_dict['Number of Qubits']/2), axis = 1))

                # Train Score and margin
                accuracy, margin_dist, margin_boxplot, margin_mean = score_and_margins(model, train_x_stacked, train_y)
                
                # Test score
                correct_predictions = 0
                with torch.no_grad():
                    probabilities = model(test_x_stacked)
                predictions = torch.argmax(probabilities, dim=1)
                correct_predictions += (predictions == test_y).sum().item()
                test_accuracy = correct_predictions / len(test_y)

                # Save
                score_margin_dict = {'train_acc': accuracy, 'train_margin_dist': list(margin_dist), 'train_margin_boxplot': list(margin_boxplot),
                                    'train_margin_mean': margin_mean, 'test_acc': test_accuracy}
                with open(os.path.join(dataset, folder_name, 'score_margin.json'), 'w') as f:
                    json.dump(score_margin_dict, f)