"""Built-in feed-forward probability model."""

import torch


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=(), negative_slope=0.01):
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        if input_size < 1:
            raise ValueError("input_size must be positive.")
        if any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")

        self.input_size = int(input_size)
        self.hidden_sizes = hidden_sizes
        self.negative_slope = float(negative_slope)
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, h))
            layers.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
            in_size = h
        layers.append(torch.nn.Linear(in_size, 1))
        layers.append(torch.nn.Sigmoid())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
