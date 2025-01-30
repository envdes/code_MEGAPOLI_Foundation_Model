"""
This script implements a fine-tuning approach for a aerosol mixing state pre-trained foundation model using Optuna for hyperparameter optimization.
It loads MEGAPOLI data, applies normalization based on PartMC data, and performs transfer learning by fine-tuning a pre-trained foundation model.
The script optimizes two hyperparameters: the number of frozen layers (10 - 14) and the L2 regularization weight (weight decay (1e-5 - 1e-2)).

Author: Fei Jiang, The University of Manchester
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import optuna


# Function to set random seed for reproducibility
def set_seed(seed):
    """
    Sets the random seed for reproducibility across numpy, random, and PyTorch.

    Parameters:
    - seed (int): The seed value to ensure reproducibility.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)  # CPU
    if torch.cuda.is_available():  # GPU
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the random seed to ensure reproducibility
seed = 42  # Seed value for reproducibility
set_seed(seed)

features = ['O3 (ppb)', 'CO (ppb)', 'NO (ppb)', 'NOx (ppb)',
        'ETH (ppb)', 'TOL(ppb)', 'XYL (ppb)', 'ALD2 (ppb)',
       'AONE (ppb)', 'PAR (ppb)', 'OLET (ppb)', 'Temperature(K)', 'RH',
       'BC (ug/m3)', 'OA (ug/m3)', 'NH4 (ug/m3)', 'NO3 (ug/m3)', 'SO4 (ug/m3)']


# Load the dataset used for training the original model for Normalizing
partmc_train_data = pd.read_csv('../Data/PartMC_data/PartMC_train.csv')
X_train = partmc_train_data[features]


# Load the fine-tuning datasets
megapoli_train_data = pd.read_csv('../Data/MEGAPOLI_data/MEGAPOLI_Marine_train_50%.csv') # e.g. Use 50% fine-tuning training dataset to fine-tune foundation model
megapoli_test_data = pd.read_csv('../Data/MEGAPOLI_data/MEGAPOLI_Marine_test_50%.csv')

def load_data(megapoli_train_data, megapoli_test_data):

    # Prepare MEGAPOLI data
    X_megapoli_train = megapoli_train_data[features]
    y_megapoli_train = megapoli_train_data.iloc[:, 23]
    X_megapoli_test = megapoli_test_data[features]
    y_megapoli_test = megapoli_test_data.iloc[:, 23]

    # Standardize the data using the scaler from the original model's training data
    scaler_X = StandardScaler()
    X_train2 = scaler_X.fit_transform(X_train)  #  Fit on the original training data, X_train2 ensures no need to reload the original dataset (PartMC)
    X_megapoli_train = scaler_X.transform(X_megapoli_train)
    X_megapoli_test = scaler_X.transform(X_megapoli_test)
    return X_megapoli_train, y_megapoli_train, X_megapoli_test, y_megapoli_test


# Define the ResNet-like model architecture
class ResNetBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_features, num_blocks, hidden_size):
        super(ResNet, self).__init__()
        self.fc_in = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.fc_in(x)
        out = self.relu(out)
        out = self.blocks(out)
        out = self.fc_out(out)
        return out

# Define the Optuna objective function for hyperparameter optimization
def objective(trial):
    X_megapoli_train, y_megapoli_train, X_megapoli_test, y_megapoli_test = load_data(megapoli_train_data, megapoli_test_data)
    
    X_megapoli_train_tensor = torch.tensor(X_megapoli_train, dtype=torch.float32)
    y_megapoli_train_tensor = torch.tensor(y_megapoli_train, dtype=torch.float32)
    X_megapoli_test_tensor = torch.tensor(X_megapoli_test, dtype=torch.float32)
    y_megapoli_test_tensor = torch.tensor(y_megapoli_test, dtype=torch.float32)

    megapoli_train_dataset = TensorDataset(X_megapoli_train_tensor, y_megapoli_train_tensor)
    train_loader = DataLoader(megapoli_train_dataset, batch_size=1, shuffle=True)
    
    input_size = X_megapoli_train_tensor.shape[1]
    num_blocks = 15
    hidden_size = 512
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-3)    
    num_frozen_blocks = trial.suggest_int("num_frozen_blocks", 10, 14)  # Optimize number of frozen layers
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)  # Optimize L2 regularization
    
    model = ResNet(input_size, num_blocks, hidden_size)
    model.load_state_dict(torch.load('../Model/Foundation_Model.pth'))
    
    # Freeze the selected number of layers
    for i, block in enumerate(model.blocks):
        if i < num_frozen_blocks:
            for param in block.parameters():
                param.requires_grad = False
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    num_epochs = 30
    best_mse = float('inf')
    no_improve = 0
    patience = 3
    best_weights = None


    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_megapoli_test_tensor).view(-1).numpy()
            y_true = y_megapoli_test_tensor.numpy()
            test_mse = mean_squared_error(y_true, predictions)

        if test_mse < best_mse:
            best_mse = test_mse
            best_weights = model.state_dict().copy()
            no_improve = 0
        else: 
            no_improve += 1
            if no_improve >= patience:
                break
    
    model.load_state_dict(best_weights)

    return best_mse

# Run Optuna hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)

# Output best hyperparameters
print("Best hyperparameters:", study.best_params)

