import numpy as np

class RMSNorm(object):
    def __init__(self, dim, epsilon=1e-8):
        self.dim = dim
        self.epsilon = epsilon
        self.weight = np.ones(dim)

    def forward(self, x):
        # x shape: (batch_size, dim)
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.epsilon)
        norm_x = x / rms  # Normalize
        return norm_x * self.weight  # Scale

    def __call__(self, x):
        return self.forward(x)