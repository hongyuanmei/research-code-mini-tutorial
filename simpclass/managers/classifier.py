# code migrated from: https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification
import torch
from torch import nn
from torch import optim 

class Classifier():
    
    def __init__(self, model, criterion=nn.CrossEntropyLoss, optimizer=optim.Adam):
        self.model = model
        self.criterion = criterion()
        self.optimizer = optimizer(self.model.parameters())
    
    def train(self, train_loader, dev_loader=None, n_epochs=10):
        # set to train mode
        self.model.train()
        for epoch in range(n_epochs):
            for feature, target in train_loader:
                # compute loss and back-prop
                self.optimizer.zero_grad()
                output = self.model(feature)                
                loss = self.criterion(output, target.long())
                loss.backward()
                self.optimizer.step()
            # check dev accuracy
            if dev_loader != None:
                acc = self.validate(dev_loader)
                print(f"{acc*100.0:.2f}% : dev acc of epoch-{epoch}")

    def validate(self, data_loader):
        self.model.eval()
        correct = 0.
        total = 0.
        with torch.no_grad():
            for feature, target in data_loader:
                output = self.model(feature)
                _, prediction = torch.max(output, dim=1)
                correct += torch.sum(prediction==target).item()
                total += target.size(0)
        return 1.0 * correct / total
        
    def test(self, data_loader): 
        acc = self.validate(data_loader)
        print(f"{acc*100.0:.2f}% : test acc")