from typing import Callable, List, Tuple, Union, cast

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from d3rlpy.models.encoders import Encoder, EncoderWithAction
from d3rlpy.models.torch.policies import Policy, squash_action
from d3rlpy.models.torch.q_functions.ensemble_q_function import _reduce_ensemble


class EnsembleDeterministicPolicy(Policy):
    _encoder: Encoder
    _fc: nn.ModuleList

    def __init__(self, encoder: Encoder, action_size: int, n_actors: int = 1):
        super().__init__()
        self.n_actors = n_actors
        self._encoder = encoder
        self._action_size = action_size
        feature_size = encoder.get_feature_size()
        self._fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    nn.ReLU(),
                    nn.Linear(feature_size, action_size),
                )
                for _ in range(self.n_actors)
            ]
        )

    def forward(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        h = self._encoder(x)
        values = []
        for fc in self._fc:
            values.append(
                torch.tanh(fc(h)).view(1, h.shape[0], self._action_size)
            )

        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, reduction))

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class EnsembleDeterministicResidualPolicy(Policy):
    _encoder: EncoderWithAction
    _scale: float
    _fc: nn.ModuleList

    def __init__(
        self, encoder: EncoderWithAction, scale: float, n_actors: int = 1
    ):
        super().__init__()
        self._scale = scale
        self._encoder = encoder
        self.n_actors = n_actors
        self._fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        encoder.get_feature_size(), encoder.get_feature_size()
                    ),
                    nn.ReLU(),
                    nn.Linear(encoder.get_feature_size(), encoder.action_size),
                )
                for _ in range(self.n_actors)
            ]
        )

    def forward(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        values = []
        for fc in self._fc:
            residual_action = self._scale * torch.tanh(fc(h))
            values.append(
                (action + cast(torch.Tensor, residual_action)).clamp(-1.0, 1.0)
            )

        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, reduction))

    def best_residual_action(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return self.forward(x, action, reduction)

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "residual policy does not support best_action"
        )

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )


class EnsembleCallableDeterministicResidualPolicy(EnsembleDeterministicPolicy):
    _policy: Callable

    def __init__(
        self,
        encoder: Encoder,
        policy: Callable,
        action_size: int,
        scale: float,
        n_actors: int = 1,
    ):
        super(EnsembleCallableDeterministicResidualPolicy).__init__(
            encoder, action_size, n_actors
        )
        self._scale = scale
        self._policy = policy

    def forward(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        policy_action = cast(torch.Tensor, self._policy(x))
        residual_action = super().forward(x, reduction)
        action = policy_action + residual_action
        action = action.clamp(-1.0, 1.0)

        return action

    def __call__(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.forward(x, reduction))

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class EnsembleSquashedNormalPolicy(Policy):  # WIP TODO
    _encoder: Encoder
    _action_size: int
    _min_logstd: float
    _max_logstd: float
    _use_std_parameter: bool
    _mu: nn.ModuleList
    _logstd: Union[nn.ModuleList, nn.ParameterList]

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
        n_actors: int = 1,
    ):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._use_std_parameter = use_std_parameter
        self.n_actors = n_actors

        self._mu = nn.ModuleList(
            [
                nn.Linear(encoder.get_feature_size(), action_size)
                for _ in range(self.n_actors)
            ]
        )
        if use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self._logstd = nn.ParameterList(
                [nn.Parameter(initial_logstd) for _ in range(self.n_actors)]
            )
        else:
            self._logstd = nn.ModuleList(
                [
                    nn.Linear(encoder.get_feature_size(), action_size)
                    for _ in range(self.n_actors)
                ]
            )

    def _compute_logstd(self, h: torch.Tensor) -> torch.Tensor:
        if self._use_std_parameter:
            clipped_logstd = self.get_logstd_parameter()
        else:
            logstd = cast(nn.Linear, self._logstd)(h)
            clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return clipped_logstd

    def dist(self, x: torch.Tensor) -> List[Normal]:
        h = self._encoder(x)
        values = []
        for mu in self._mu:
            clipped_logstd = self._compute_logstd(h)
            values.append(Normal(mu(h), clipped_logstd.exp()))

        return values

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
        reduction: str = "mean",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if deterministic:
            # to avoid errors at ONNX export because broadcast_tensors in
            # Normal distribution is not supported by ONNX

            h = self._encoder(x)
            values = []
            for mu in self._mu:
                action = mu(h)
                values.append(action)

            return _reduce_ensemble(torch.cat(values, dim=0), reduction)
        else:
            dist = self.dist(x)
            values = [d.rsample() for d in dist]
            action = _reduce_ensemble(torch.cat(values, dim=0), reduction)

        if with_log_prob:
            actions = [squash_action(d, v) for d, v in zip(dist, values)]
            squashed_action = _reduce_ensemble(
                torch.cat([a[0] for a in actions], dim=0), reduction
            )
            log_prob = _reduce_ensemble(
                torch.cat([a[0] for a in actions], dim=0), reduction
            )
            return squashed_action, log_prob

        return torch.tanh(action)

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)

        action = dist.rsample((n,))

        squashed_action, log_prob = squash_action(dist, action)

        # (n, batch, action) -> (batch, n, action)
        squashed_action = squashed_action.transpose(0, 1)
        # (n, batch, 1) -> (batch, n, 1)
        log_prob = log_prob.transpose(0, 1)

        return squashed_action, log_prob

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        dist = self.dist(x)
        action = dist.rsample((n,))
        return action.transpose(0, 1)

    def onnx_safe_sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        h = self._encoder(x)
        mean = self._mu(h)
        std = self._compute_logstd(h).exp()

        # expand shape
        # (batch_size, action_size) -> (batch_size, N, action_size)
        expanded_mean = mean.view(-1, 1, self._action_size).repeat((1, n, 1))
        expanded_std = std.view(-1, 1, self._action_size).repeat((1, n, 1))

        # sample noise from Gaussian distribution
        noise = torch.randn(x.shape[0], n, self._action_size, device=x.device)

        return torch.tanh(expanded_mean + noise * expanded_std)

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        action = self.forward(x, deterministic=True, with_log_prob=False)
        return cast(torch.Tensor, action)

    def get_logstd_parameter(self) -> torch.Tensor:
        assert self._use_std_parameter
        logstd = torch.sigmoid(cast(nn.Parameter, self._logstd))
        base_logstd = self._max_logstd - self._min_logstd
        return self._min_logstd + logstd * base_logstd


class EnsembleCategoricalPolicy(Policy):  # WIP TODO
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x: torch.Tensor) -> Categorical:
        h = self._encoder(x)
        h = self._fc(h)
        return Categorical(torch.softmax(h, dim=1))

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dist = self.dist(x)

        if deterministic:
            action = cast(torch.Tensor, dist.probs.argmax(dim=1))
        else:
            action = cast(torch.Tensor, dist.sample())

        if with_log_prob:
            return action, dist.log_prob(action)

        return action

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)

        action_T = cast(torch.Tensor, dist.sample((n,)))
        log_prob_T = dist.log_prob(action_T)

        # (n, batch) -> (batch, n)
        action = action_T.transpose(0, 1)
        # (n, batch) -> (batch, n)
        log_prob = log_prob_T.transpose(0, 1)

        return action, log_prob

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.forward(x, deterministic=True))

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        return cast(torch.Tensor, dist.logits)
