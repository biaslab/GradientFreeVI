import torch
import hyperopt

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from bayes_opt import BayesianOptimization
import copy
import numpy as np
import random
random.seed(123)


training_data = datasets.MNIST(
    root="pytorch experiments/data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="pytorch experiments/data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_data, validation_data = torch.utils.data.random_split(test_data, [9000, 1000])

train_dataloader = DataLoader(training_data, batch_size=128)
validation_dataloader = DataLoader(validation_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train_model(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train the model
    for t in tqdm(range(20)):
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def run_model(learning_rate):
    # initialize model
    model = NeuralNetwork()
    model.to(device)


    # your training here
    # define the loss function and the optimizer
    train_model(model, learning_rate)
    # compute test accuracy
    with torch.no_grad():
        test_accuracy = 0
        for images, labels in tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_accuracy += (outputs.argmax(1) == labels).float().mean().item()
        test_accuracy /= len(test_dataloader)
    return test_accuracy

import json

with open("rxinfer_results_lrs.json", "r") as file:
    lrs = json.load(file)

with open("rxinfer_results.json", "r") as file:
    results = json.load(file)

with open("bayesopt_results.json", "r") as file:
    bayesopt_results = json.load(file)

for distribution, learned_lrs in lrs[1].items():
    print(distribution)
    print(run_model(learned_lrs))

for method, learned_lr in bayesopt_results["params"].items():
    if learned_lr < 0:
        learned_lr = np.exp(learned_lr)
    print(method)
    print(run_model(learned_lr))

def run_ensemble(lrs):
    models = []
    for lr in lrs:
        model = NeuralNetwork()
        model.to(device)
        model = train_model(model, lr)
        models.append(model)
    with torch.no_grad():
        # run voting ensemble over trained models
        test_accuracy = 0
        for images, labels in tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = torch.stack([model(images) for model in models])
            outputs = outputs.mean(0)
            test_accuracy += (outputs.argmax(1) == labels).float().mean().item()
        test_accuracy /= len(test_dataloader)
    return test_accuracy

#for parameters in lrs[0]:
#   print(run_ensemble(parameters))