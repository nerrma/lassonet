import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# Proximal algorithm detailed in next section
from prox import inplace_prox
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, scale

import matplotlib.pyplot as plt

# Define a general Lasso Net neural network architecture using Pytorch
class GeneralModel(nn.Module):
    def __init__(self, dims=[10, 10]):
       super().__init__()

       # Arbitrary feed-forward neural net architecture based on dims 
       self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

       # Skip layer for LassoNet
       self.skip_layer = nn.Linear(dims[0], dims[-1], bias=False)

    # Forward using residual mapping
    def forward(self, inputs):
        current_layer = inputs
        result = self.skip_layer(inputs)
        
        # Run residual
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                current_layer = F.relu(current_layer)
        return result + current_layer
        
    def prox(self, *, lambda_, lambda_bar=0, M=1):
       with torch.no_grad():
           inplace_prox(
               beta=self.skip_layer,
               theta=self.layers[0],
               lambda_=lambda_,
               lambda_bar=lambda_bar,
               M=M,
           )

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}

# Define pytorch boilerplate
class LassoNetModel():
    def __init__(self, dims=None):
        self.model = GeneralModel(dims=dims)
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.M = 30
        self.device = torch.device("cpu:0")
        self.epsilon = 0.02

    def _cast_input(self, X, y=None):
        X = torch.FloatTensor(X).to(self.device)
        if y is None:
            return X
        y = self._convert_y(y)
        return X, y

    def load(self, state_dict):
        self.model.load_state_dict(state_dict)
        
    def train_lassonet(self, X_train, y_train, epochs):
        # Ensure inputs are Pytorch tensors
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train)
        n_train = len(X_train)
        lambda_ = 0.1


        # Init simple optimiser
        optimiser = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)
        k = len(X_train)
        i = 0
        path = []

        # Main loop, terminate when no features used or 1000 iterations
        # Refer to psudeocode for structure
        while k > 0 and i < 1000:
            lambda_ = (1+self.epsilon)*lambda_
            for epoch in range(epochs):
                   self.model.train()
                   loss = 0
                   def closure():
                       nonlocal loss
                       optimiser.zero_grad()
                       ans = self.criterion(self.model(X_train), y_train.unsqueeze(1))
                       ans.backward()
                       loss += ans.item() / n_train
                       return ans

                   optimiser.step(closure)
                   self.model.prox(
                       lambda_=lambda_ * optimiser.param_groups[0]["lr"], M=self.M
                       )
            k = np.count_nonzero(self.model.skip_layer.weight.detach().numpy())
            i += 1

            # Save state dict for path
            state_dict = self.model.cpu_state_dict()
            path.append(state_dict)

            if i%10 == 0:
                print(f"Iteration {i}, k = {k}, loss = {loss}, lbd = {lambda_}")
        return path
        
    def predict(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

# Sample train
X, y = load_boston(return_X_y=True)

X = StandardScaler().fit_transform(X)
y = scale(y)

# Naive split dataset
X_train, y_train = X[:100], y[:100] 
X_val, y_val = X[101:200], y[101:200] 
X_test, y_test = X[201:250], y[201:250] 

# Train model and get path
ls = LassoNetModel([len(X_train[0]), 8, 16, 16, 32, 64, 1])
path = ls.train_lassonet(X_train, y_train, 50)

# Find best model
best = path[0]
best_mse = np.inf 
for p in path:
    ls.load(p)
    y_preds = ls.predict(X_val)
    mse = mean_squared_error(y_preds, y_val)
    if mse < best_mse:
        best = p
        best_mse = mse

# Load best model and plot test results
ls.load(best)
y_preds = ls.predict(X_test)
mse = mean_squared_error(y_preds, y_test)

plt.plot(y_preds, c='r')
plt.plot(y_test)
plt.show()
print(f"test mse = {mse}")
