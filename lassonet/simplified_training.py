import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# Proximal algorithm detailed in next section
from prox import inplace_prox

class GeneralModel(nn.Module):
    def __init__(self, dims=[10, 10]):
       super().__init__()
       # Arbitrary feed-forward neural net architecture based on dims 
       self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
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

# Define pytorch boilerplate
class LassoNetModel():
    def __init__(self, dims=None):
        self.model = GeneralModel(dims=dims)
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.M = 30

    def train_lassonet(self, X_train, y_train, epochs):
        n_train = len(X_train)
        lambda_ = 0

        optimiser = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)
        k = len(X_train)
        i = 0
        while k > 0:
            for epoch in range(epochs):
                   self.model.train()
                   loss = 0
                   for i in range(n_train):
                       def closure():
                           nonlocal loss
                           optimiser.zero_grad()
                           ans = self.criterion(self.model(X_train), y_train)
                           ans.backward()
                           loss += ans.item() / n_train
                           return ans

                       optimiser.step(closure)
                       self.model.prox(
                           lambda_=lambda_ * optimiser.param_groups[0]["lr"], M=self.M
                       )
            k = np.count_nonzero(self.model.skip_layer.weight.detach().numpy())
            i += 1
            print(f"Iteration {i}, k = {k}")
