import os
import torch
from simpclass.models.mlp import MLP
from simpclass.managers.classifier import Classifier
from utils import load_data, save_model

def main():

    # FIX RANDOM SEED!!!
    torch.random.manual_seed(123)
    
    data_path = '../../data/gaussian'
    batch_size = 128

    train_loader = load_data(data_path, 'train', batch_size)
    dev_loader = load_data(data_path, 'dev', batch_size)

    mymodel = MLP()
    myclassifier = Classifier(mymodel)
    myclassifier.train(train_loader, dev_loader)

    # save the trained model 
    save_path = '../../logs/gaussian'
    save_model(mymodel, save_path, f'BS={batch_size}')


if __name__ == "__main__": main()