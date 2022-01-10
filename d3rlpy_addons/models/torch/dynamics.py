# pylint: disable=protected-access

from typing import List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import spectral_norm

from d3rlpy.models.encoders import EncoderWithAction
from d3rlpy.models.torch.dynamics import _apply_spectral_norm_recursively, _compute_ensemble_variance, _gaussian_likelihood


class StateProbabilisticDynamicsModel(nn.Module):  # type: ignore
    """Probabilistic dynamics model.
    """

    _encoder: EncoderWithAction
    _mu: nn.Linear
    _logstd: nn.Linear
    _max_logstd: nn.Parameter
    _min_logstd: nn.Parameter

    def __init__(self, encoder: EncoderWithAction):
        super().__init__()
        # apply spectral normalization except logstd encoder.
        _apply_spectral_norm_recursively(cast(nn.Module, encoder))
        self._encoder = encoder

        feature_size = encoder.get_feature_size()
        observation_size = encoder.observation_shape[0]
        out_size = observation_size

        # TODO: handle image observation
        self._mu = spectral_norm(nn.Linear(feature_size, out_size))
        self._mu = nn.Linear(feature_size, out_size)
        self._logstd = nn.Linear(feature_size, out_size)

        # logstd bounds
        init_max = torch.empty(1, out_size, dtype=torch.float32).fill_(2.0)
        init_min = torch.empty(1, out_size, dtype=torch.float32).fill_(-10.0)
        self._max_logstd = nn.Parameter(init_max)
        self._min_logstd = nn.Parameter(init_min)

    def compute_stats(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder(x, action)

        mu = self._mu(h)

        # log standard deviation with bounds
        logstd = self._logstd(h)
        logstd = self._max_logstd - F.softplus(self._max_logstd - logstd)
        logstd = self._min_logstd + F.softplus(logstd - self._min_logstd)

        return mu, logstd

    def forward(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_variance(x, action)[:2]

    def predict_with_variance(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.compute_stats(x, action)
        dist = Normal(mu, logstd.exp())
        pred = dist.rsample()
        # residual prediction
        next_x = x + pred
        next_reward = pred.view(-1, 1)*0.0
        return next_x, next_reward, dist.variance.sum(dim=1, keepdims=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> torch.Tensor:
        mu, logstd = self.compute_stats(obs_t, act_t)

        # residual prediction
        mu_x = obs_t + mu
        logstd_x = logstd

        # gaussian likelihood loss
        likelihood_loss = _gaussian_likelihood(obs_tp1, mu_x, logstd_x)

        # penalty to minimize standard deviation
        penalty = logstd_x.sum(dim=1, keepdim=True)

        # minimize logstd bounds
        bound_loss = self._max_logstd.sum() - self._min_logstd.sum()

        loss = likelihood_loss + penalty + 1e-2 * bound_loss

        return loss.view(-1, 1)

