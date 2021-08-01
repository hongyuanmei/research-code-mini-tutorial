import os
import torch
from simpclass.models.mlp import MLP
from simpclass.managers.classifier import Classifier
from utils import load_data, load_model

def main():

    # FIX RANDOM SEED!!!
    torch.random.manual_seed(123)
    
    data_path = '../../data/gaussian'
    batch_size = 128

    test_loader = load_data(data_path, 'test', batch_size)

    # load the saved model 
    mymodel = load_model(MLP, data_path, f'BS={batch_size}')
    
    myclassifier = Classifier(mymodel)
    myclassifier.test(test_loader)


if __name__ == "__main__": main()