import os
from datetime import datetime
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_path, split, batch_size):
    # read raw data
    with open(f'{data_path}/{split}.pkl', 'rb') as f: 
        X, Y = pickle.load(f)
    # organize data into torch dataloaders
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    data_loader = DataLoader(dataset, batch_size)
    return data_loader

def save_model(model, model_path, attr): 
    torch.save(model.state_dict(), f'{model_path}/model-{attr}.pt')

def load_model(model_class, model_path, attr): 
    model = model_class()
    model.load_state_dict(torch.load(f'{model_path}/model-{attr}.pt'))
    return model