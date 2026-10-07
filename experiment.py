import pennylane as qml
import torch
import os
import random
from torchvision.datasets import KMNIST, MNIST, FashionMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def build_model_from_hyperparameter_dict(hyperparameter_dict):
    """
    Given a dictionary of hyperparameters, create corresponding torch module

    Args:
        hyperparameter_dict (dict): Dictionary with keys ['Number of Qubits', 'Number of ReUps', 'Embedding', 'Ansatz', 'Random Seed']
    Returns:
        model (torch.nn.Module)
    """
    num_qubits = hyperparameter_dict['Number of Qubits']
    num_reups = hyperparameter_dict['Number of ReUps']
    
    # Parse embedding
    embedding_type, embedding_setting = hyperparameter_dict['Embedding'].split('/')
    if embedding_type == 'AngleEmbedding':
        rotation_type = embedding_setting[1]
        def embedding(x):
            qml.AngleEmbedding(x, wires = range(num_qubits), rotation = rotation_type)
    elif embedding_type == 'IQPEmbedding':
        rotation_layer = int(embedding_setting[0])
        def embedding(x):
            qml.IQPEmbedding(x, wires = range(num_qubits), n_repeats = rotation_layer, pattern = [[i, i+1] for i in range(num_qubits-1)])
    else:
        raise ValueError('Unsupported embedding')

    # Parse ansatz
    ansatz_type, ansatz_setting = hyperparameter_dict['Ansatz'].split('/')
    if ansatz_type == 'BasicEntanglerLayers':
        if ansatz_setting == 'RX':
            rotation_gate = qml.RX
        elif ansatz_setting == 'RY':
            rotation_gate = qml.RY
        elif ansatz_setting == 'RZ':
            rotation_gate = qml.RZ
        else:
            raise ValueError
        def ansatz(weights):
            qml.BasicEntanglerLayers(weights, wires = range(num_qubits), rotation = rotation_gate)
        weight_shapes = {"weights": (num_reups, 1, num_qubits)}
    elif ansatz_type == 'StronglyEntanglingLayers':
        ansatz_layer = int(ansatz_setting[0])
        def ansatz(weights):
            qml.StronglyEntanglingLayers(weights, wires = range(num_qubits))
        weight_shapes = {"weights": (num_reups, ansatz_layer, num_qubits, 3)}
    elif ansatz_type == 'SimplifiedTwoDesign':
        ansatz_layer = int(ansatz_setting[0])
        def ansatz(init_weights, weights):
            qml.SimplifiedTwoDesign(init_weights, weights, wires = range(num_qubits))
        weight_shapes = {"init_weights": (num_reups, num_qubits), "weights": (num_reups, ansatz_layer, num_qubits-1, 2)}
    else:
        raise ValueError('Unsupported ansatz')

    # Construct model
    dev = qml.device("default.qubit", wires=num_qubits)
    if 'init_weights' in weight_shapes.keys():
        @qml.qnode(dev, interface = 'torch')
        def classifier(inputs, init_weights, weights):
            for l in range(num_reups):
                embedding(inputs)
                ansatz(init_weights[l], weights[l])
            return qml.probs(wires = [0])
    else:
        @qml.qnode(dev, interface = 'torch')
        def classifier(inputs, weights):
            for l in range(num_reups):
                embedding(inputs)
                ansatz(weights[l])
            return qml.probs(wires = [0])
        
    class model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(classifier, weight_shapes)
        def forward(self, x):
            return self.qlayer(x)
    torch.manual_seed(hyperparameter_dict['Random Seed'])
    model = model()
    return model

def preprocess_data(dataset, classes, n_features, n_train, n_test):
    """
    Preprocess data for binary classification.
    
    Args:
        dataset (str): 'mnist'/'kmnist'/'fashion'
        classes (list[int]): classes used for binary classification
        n_features (int): number of principal components to keep
        n_train: number of training samples
        n_test: number of testing samples

    Returns:
        train_x, train_y, test_x, test_y
    """
    if dataset == 'mnist':
        train_ds = MNIST(root = './', train = True, download=True)
        test_ds = MNIST(root = './', train = False, download=True)
    elif dataset == 'kmnist':
        train_ds = KMNIST(root = './', train = True)
        test_ds = KMNIST(root = './', train = False)
    elif dataset == 'fashion':
        train_ds = FashionMNIST(root = './', train = True, download=True)
        test_ds = FashionMNIST(root = './', train = False, download=True)

    # Filter
    train_ds_classes = [i for i in train_ds if (i[1] == classes[0] or i[1] == classes[1])] 
    test_ds_classes = [i for i in test_ds if (i[1] == classes[0] or i[1] == classes[1])]

    # Balance dataset
    train_labels = np.array([i[1] for i in train_ds_classes])
    train_class0_indices = np.argwhere(train_labels == classes[0])
    train_class1_indices = np.argwhere(train_labels == classes[1])
    np.random.shuffle(train_class0_indices)
    np.random.shuffle(train_class1_indices)
    train_balanced_indices = np.append(train_class0_indices[:int(n_train/2)], train_class1_indices[:int(n_train/2)])
    np.random.shuffle(train_balanced_indices)

    test_labels = np.array([i[1] for i in test_ds_classes])
    test_class0_indices = np.argwhere(test_labels == classes[0])
    test_class1_indices = np.argwhere(test_labels == classes[1])
    np.random.shuffle(test_class0_indices)
    np.random.shuffle(test_class1_indices)
    test_balanced_indices = np.append(test_class0_indices[:int(n_test/2)], test_class1_indices[:int(n_test/2)])
    np.random.shuffle(test_balanced_indices)

    # Separate labels from data and change classes to 0 or 1
    train_y = np.array([train_ds_classes[i][1] for i in train_balanced_indices])
    train_y = torch.tensor(train_y == classes[1]).int()
    test_y = np.array([test_ds_classes[i][1] for i in test_balanced_indices])
    test_y = torch.tensor(test_y == classes[1]).int()

    # Separate data from labels
    train_x = np.array([np.array(train_ds_classes[i][0]) for i in train_balanced_indices])
    test_x = np.array([np.array(test_ds_classes[i][0]) for i in test_balanced_indices])

    # final number of features (between 1 and 28x28)
    train_X_flat = np.reshape(train_x, (-1, 28 * 28))
    test_X_flat = np.reshape(test_x, (-1, 28 * 28))

    # Standardize and PCA
    scaler = StandardScaler()
    train_X_flat = scaler.fit_transform(train_X_flat)
    test_X_flat = scaler.transform(test_X_flat)

    pca = PCA(n_components=n_features)
    train_x = torch.tensor(pca.fit_transform(train_X_flat))
    test_x = torch.tensor(pca.transform(test_X_flat))

    # Standardize again
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return train_x, train_y, test_x, test_y

def train(x_train, y_train, lr, batchsize, qlayer, save_path, seed, convergence_interval = 500):
    """
    Train model

    Args:
        x_train: train data
        y_train: train labels
        lr: learning rate
        batchsize: batch size
        qlayer: model to train
        save_path: path to save checkpoints and other data
        seed: random seed
        convergence_interval: number of minibatch updates to use for convergence check
    """
    # Set random seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    batch_loss_history = []
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(qlayer.parameters(), lr = lr)

    train_ds = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = batchsize, shuffle=True)

    step = 0
    while True:
        for batch_x, batch_y in train_dl:
            opt.zero_grad()
            outputs = qlayer(batch_x)
            loss_value = loss(outputs, batch_y.long())
            loss_value.backward()
            opt.step()

            batch_loss_history.append(loss_value.item())

            if step % 1000 == 0:
                print(f"Step {step}, Loss {loss_value.item()}")

            # Check convergence
            if step > 2 * convergence_interval:
                average1 = np.mean(batch_loss_history[-convergence_interval:])
                average2 = np.mean(batch_loss_history[-2 * convergence_interval:-convergence_interval])
                std1 = np.std(batch_loss_history[-convergence_interval:])
                if np.abs(average1 - average2) < std1 / np.sqrt(convergence_interval) / 2:
                    print(f"Convergence reached at step {step}")

                    # Save
                    np.save(os.path.join(save_path, "loss_history"), np.array(batch_loss_history)) # Loss history
                    torch.save(qlayer.state_dict(), os.path.join(save_path, "model_params.pth")) # Model parameters
                    return
            step += 1

def score_and_margins(model, X, y):
    """
    Score traied models
    
    Args:
        model (torch.nn.Module): trained model
        X: train data
        y: train labels

    Returns:
        accuracy, margin distribution, margin boxplot, margin mean
    """
    correct_predictions = 0
    margin_dists = []

    # Get probabilities for the batch
    with torch.no_grad():
        probabilities = model(X)
            
    # Score calculation
    predictions = torch.argmax(probabilities, dim=1)
    correct_predictions += (predictions == y).sum().item()
    accuracy = correct_predictions / len(y)

    # Margin calculation
    correct_label_probs = probabilities.gather(1, y.view(-1, 1)).squeeze(1)
    incorrect_label_probs, _ = torch.max(probabilities.masked_fill(torch.eye(probabilities.size(1))[y].bool(), float('-inf')), dim=1)
    margin_dist = correct_label_probs - incorrect_label_probs

    # Calculate Margin Mean
    margin_mean = margin_dist.mean().item()
    
    # Zero out margins for incorrect predictions
    margin_dist = torch.where(predictions == y, margin_dist, torch.zeros_like(margin_dist))
    margin_dists.append(margin_dist)

    # Calculate margin statistics
    margin_dist = torch.cat(margin_dists, dim=0) 
    margin_min = margin_dist.min().item()
    margin_Q1 = torch.quantile(margin_dist, 0.25).item()
    margin_Q2 = torch.quantile(margin_dist, 0.50).item()
    margin_Q3 = torch.quantile(margin_dist, 0.75).item()
    margin_max = margin_dist.max().item()
    margin_boxplot = np.array([margin_min, margin_Q1, margin_Q2, margin_Q3, margin_max])

    return accuracy, margin_dist.detach().numpy(), margin_boxplot, margin_mean