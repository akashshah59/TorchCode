import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        """Normal equation: w = (X^T X)^{-1} X^T y"""
        # X = (B,C)
        # C X B , B X C = C X C X C X B = > C X B * B
        
        X_aug = torch.concat([X, torch.ones(X.shape[0],1)], dim = -1)
        
        w = torch.inverse(X_aug.T @ X_aug) @ X_aug.T @ y

        return (w[:-1],w[-1])

    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor,
                         lr: float = 0.01, steps: int = 1000):
        """Manual gradient descent loop"""

        # X = (B x D)
        # Y = (B,)

        w = torch.zeros(X.shape[1])
        b = torch.zeros(1)

        while steps > 0:
            pred = X @ w + b # B * D X D => B + bias to all. 
            error = pred - y
            grad_w = (2/ X.shape[0]) * X.T @ error # D * B , B X 1 => D , 1 . gradients along all tensors
            grad_b = (2 / X.shape[0]) * error.sum() 
            w -= lr * grad_w
            b -= lr * grad_b
            steps -= 1

        return (w,b)

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        """Train nn.Linear with autograd"""

        linear = nn.Linear(X.shape[1], 1)
        optimizer = torch.optim.Adam(linear.parameters(), lr = lr)
        mse_loss = nn.MSELoss()
        
        while steps > 0:
            optimizer.zero_grad()
            y_hat = linear(X)
            # print(y_hat.shape)
            # print(y_hat.squeeze(-1).shape)
            loss = mse_loss(y_hat.squeeze(-1), y)
            loss.backward()
            optimizer.step()
            if steps % 100 == 0:
                print(loss.item())
            steps -=1
    
        w = linear.weight.data.squeeze(0)  # (D,)
        b = linear.bias.data.squeeze(0)    # scalar ()
        # return linear.parameters()
        return w, b