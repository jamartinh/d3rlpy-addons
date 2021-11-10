import torch as torch
from torch import nn


class GaussianNoiseInverseWeightLayer(nn.Module):
    def __init__(self, n_samples=5, mean=0.0, std=0.125):
        """description

        Args:
            param0 (type): description
            param1 (type, optional): description
        """
        super().__init__()

        self.n_samples = n_samples
        self.mean = mean
        self.std = std

    def forward(self, obs):

        return gaussian_noise_inverse_weight(
            obs, self.n_samples, self.mean, self.std
        )


def gaussian_noise_inverse_weight(obs, n_samples, mean, std):
    obs_repeated = torch.repeat_interleave(obs, repeats=n_samples, dim=0)
    noise_t = torch.normal(mean, std, size=obs_repeated.size()).to("cuda")
    noise_t[0 : obs.shape[0]] = 0.0
    obs_noise = obs_repeated + noise_t

    # Compute euclidean distance
    d = ((obs_repeated - obs_noise) ** 2).sum(axis=1)
    d = d.view(n_samples, obs.shape[0])

    w = 1 / (1 + d)
    w = torch.softmax(w, dim=0)
    w -= w.min()
    w /= w.max()
    w = w.view(obs_noise.shape[0], 1)
    return obs_noise, w
