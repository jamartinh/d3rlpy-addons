import copy
from typing import Union, cast

import torch
import torch.nn.functional as F
from d3rlpy.models.torch import Encoder, EncoderWithAction
from d3rlpy.models.torch.q_functions.mean_q_function import (
    ContinuousMeanQFunction,
    DiscreteMeanQFunction,
)
from d3rlpy.models.torch.q_functions.qr_q_function import (
    ContinuousQRQFunction,
    DiscreteQRQFunction,
)
from d3rlpy.models.torch.q_functions.utility import compute_reduce
from torch import nn

from d3rlpy_addons.models.torch.layers import GaussianNoiseInverseWeightLayer


class DiscreteDQRQFunction(DiscreteQRQFunction):
    _fc0: nn.Linear
    _q_value_offset: float

    def __init__(
            self,
            encoder: Encoder,
            action_size: int,
            n_quantiles: int,
            q_value_offset: float = 0.0,
    ):
        super().__init__(encoder, action_size, n_quantiles)

        # initial q_values for approximation
        self._q_value_offset = q_value_offset

        # get a new instance or clone a frozen copy
        self._fc0 = type(self._fc)(encoder.get_feature_size(), n_quantiles)

        # copy weights and stuff
        self._fc0.load_state_dict(self._fc.state_dict())

        # freeze model by freezing parameters
        for param in self._fc0.parameters():
            param.requires_grad = False

        # set fc0 in eval mode only
        self._fc0.eval()

    def _compute_quantiles(self, h: torch.Tensor,
                           taus: torch.Tensor) -> torch.Tensor:
        h = cast(torch.Tensor,
                 (self._fc(h) - self._fc0(h)) + self._q_value_offset)
        return h.view(-1, self._action_size, self._n_quantiles)


class ContinuousDQRQFunction(ContinuousQRQFunction):
    _fc0: nn.Linear
    _q_value_offset: float

    def __init__(
            self,
            encoder: EncoderWithAction,
            n_quantiles: int,
            q_value_offset: float = 0.0,
    ):
        super().__init__(encoder, n_quantiles)

        # initial q_values for approximation
        self._q_value_offset = q_value_offset

        # get a new instance or clone a frozen copy
        self._fc0 = type(self._fc)(encoder.get_feature_size(),
                                   self._n_quantiles)

        # copy weights and stuff
        self._fc0.load_state_dict(self._fc.state_dict())

        # freeze model by freezing parameters
        for param in self._fc0.parameters():
            param.requires_grad = False

        # set fc0 in eval mode only
        self._fc0.eval()

    def _compute_quantiles(self, h: torch.Tensor,
                           taus: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor,
                    (self._fc(h) - self._fc0(h)) + self._q_value_offset)


class DiscreteDMeanQFunction(DiscreteMeanQFunction):
    _fc0: nn.Linear
    _q_value_offset: float

    def __init__(self,
                 encoder: Encoder,
                 action_size: int,
                 q_value_offset: float = 0.0):
        super().__init__(encoder=encoder, action_size=action_size)

        # initial q_values for approximation
        self._q_value_offset = q_value_offset

        # get a new instance or clone a frozen copy
        self._fc0 = copy.deepcopy(self._fc)

        # copy weights and stuff
        # self._fc0.load_state_dict(self._fc.state_dict())

        # freeze model by freezing parameters
        for param in self._fc0.parameters():
            param.requires_grad = False

        # set fc0 in eval mode only
        self._fc0.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._fc(self._encoder(x)) - self._fc0(self._encoder(x)) +
            self._q_value_offset,
        )


class ContinuousDMeanQFunction(ContinuousMeanQFunction):
    _fc0: nn.Linear
    _q_value_offset: float
    _encoder0: Union[EncoderWithAction, nn.Module]

    def __init__(self,
                 encoder: EncoderWithAction,
                 q_value_offset: float = 0.0):
        super().__init__(encoder=encoder)

        # initial q_values for approximation
        self._q_value_offset = q_value_offset

        # get a new instance or clone a frozen copy
        self._fc0 = type(self._fc)(encoder.get_feature_size(), 1)
        self._encoder0 = copy.deepcopy(self._encoder)

        # # copy weights and stuff
        self._fc0.load_state_dict(self._fc.state_dict())
        self._encoder0.load_state_dict(self._encoder.state_dict())

        # freeze model by freezing parameters
        for param in self._fc0.parameters():
            param.requires_grad = False

        for param in self._encoder0.parameters():
            param.requires_grad = False

        # set fc0 in eval mode only
        self._fc0.eval()
        self._encoder0.eval()
        # self._gaussian_layer = GaussianNoiseInverseWeightLayer(n_samples=10,
        #                                                        mean=0.0,
        #                                                        std=0.01)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor,
                    self._q_value_offset + self._fc(self._encoder(x, action)) - self._fc0(self._encoder0(x, action)))

    def compute_error(self, observations: torch.Tensor,
                      actions: torch.Tensor,
                      rewards: torch.Tensor,
                      target: torch.Tensor,
                      terminals: torch.Tensor,
                      gamma: float = 0.99,
                      reduction: str = "mean") -> torch.Tensor:

        # with torch.no_grad():
        #     observations, w = self._gaussian_layer(observations)  # 64,5  ->  192,5
        #
        #     # resize act_t  64,1  ->  192,1
        #     actions = torch.repeat_interleave(
        #         actions, repeats=self._gaussian_layer.n_samples, dim=0)
        #
        # q_t = self.forward(observations, actions)
        #
        # y = rewards + gamma * target * (1 - terminals)
        # # resize y 64,1  ->  192,1
        # y = torch.repeat_interleave(y,
        #                             repeats=self._gaussian_layer.n_samples,
        #                             dim=0)
        #
        # loss = w * F.mse_loss(q_t, y, reduction="none")
        value = self.forward(observations, actions)
        y = rewards + gamma * target * (1 - terminals)
        loss = F.mse_loss(value, y, reduction="none")
        return compute_reduce(loss, reduction)
