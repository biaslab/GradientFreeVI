import torch
import hyperopt

import os
import numpy as np
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
import random
random.seed(123)


training_data = datasets.MNIST(
    root="pytorch experiments/data",
    train=True,
    download=True,
    transform=ToTensor()
)

training_data = torch.utils.data.Subset(training_data, range(3000))

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

loss_values = []

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
    
def run_model(learning_rate, exp=True):
    # initialize model
    model = NeuralNetwork()
    model.to(device)


    # your training here
    # define the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    lr = np.exp(learning_rate) if exp else learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train the model
    for t in range(20):
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # compute test accuracy
    with torch.no_grad():
        test_loss = 0
        test_accuracy = 0
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += loss_fn(outputs, labels)
            test_accuracy += (outputs.argmax(1) == labels).float().mean()
        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)
        # print(test_accuracy)
    # print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    loss_values.append(test_loss.item())
    return -test_loss.item()
losses = {}
params = {}
distributions = {}
for kernel in [RBF(), Matern(1), Matern(2.5)]:
    loss_values = [] 
    pbounds = {'learning_rate': (-7, 0)}
    optimizer = BayesianOptimization(
        f=run_model,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.set_gp_params(kernel=kernel)

    optimizer.maximize(
        init_points=2,
        n_iter=2,
    )
    losses[str(kernel)] = copy.deepcopy(loss_values)
    params[str(kernel)] = float(optimizer.max['params']['learning_rate'])
    mu, sigma = optimizer._gp.predict([[params[str(kernel)]]], return_std=True)
    distributions[str(kernel)] = (mu.item(), sigma.item())



from hyperopt import hp, tpe, fmin
loss_values = []
# Single line bayesian optimization of polynomial function
best = fmin(fn=lambda x: -run_model(x, exp=False),
            space=hp.lognormal('x', -5, 3.0),
            algo=tpe.suggest, 
            max_evals=2)

losses['TPE'] = copy.deepcopy(loss_values)
params['TPE'] = best["x"]

import json

with open('bayesopt_results.json', 'w') as f:
    json.dump({'losses': losses, 'params': params, 'distributions': distributions}, f)
    
