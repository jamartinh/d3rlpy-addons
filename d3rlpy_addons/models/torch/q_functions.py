from typing import cast

import torch
from torch import nn

from d3rlpy.models.torch import Encoder, EncoderWithAction
from d3rlpy.models.torch.q_functions.mean_q_function import ContinuousMeanQFunction, DiscreteMeanQFunction
from d3rlpy.models.torch.q_functions.qr_q_function import ContinuousQRQFunction, DiscreteQRQFunction


class DiscreteDQRQFunction(DiscreteQRQFunction):
    _fc0: nn.Linear
    _q_value_offset: float

    def __init__(self, encoder: Encoder,
                 action_size: int,
                 n_quantiles: int,
                 q_value_offset: float = 0.0
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

    def _compute_quantiles(
            self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        h = cast(torch.Tensor, (self._fc(h) - self._fc0(h)) + self._q_value_offset)
        return h.view(-1, self._action_size, self._n_quantiles)


class ContinuousDQRQFunction(ContinuousQRQFunction):
    _fc0: nn.Linear
    _q_value_offset: float

    def __init__(self, encoder: EncoderWithAction,
                 n_quantiles: int,
                 q_value_offset: float = 0.0
                 ):
        super().__init__(encoder, n_quantiles)

        # initial q_values for approximation
        self._q_value_offset = q_value_offset

        # get a new instance or clone a frozen copy
        self._fc0 = type(self._fc)(encoder.get_feature_size(), self._n_quantiles)

        # copy weights and stuff
        self._fc0.load_state_dict(self._fc.state_dict())

        # freeze model by freezing parameters
        for param in self._fc0.parameters():
            param.requires_grad = False

    def _compute_quantiles(
            self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, (self._fc(h) - self._fc0(h)) + self._q_value_offset)


class DiscreteDMeanQFunction(DiscreteMeanQFunction):  # type: ignore
    _fc0: nn.Linear
    _q_value_offset: float

    def __init__(self, encoder: Encoder, action_size: int, q_value_offset: float = 0.0):
        super().__init__(encoder=encoder, action_size=action_size)

        # initial q_values for approximation
        self._q_value_offset = q_value_offset

        # get a new instance or clone a frozen copy
        self._fc0 = type(self._fc)(encoder.get_feature_size(), 1)

        # copy weights and stuff
        self._fc0.load_state_dict(self._fc.state_dict())

        # freeze model by freezing parameters
        for param in self._fc0.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x)) - self._fc0(self._encoder(x)) + self._q_value_offset)


class ContinuousDMeanQFunction(ContinuousMeanQFunction):  # type: ignore
    _fc0: nn.Linear
    _q_value_offset: float

    def __init__(self, encoder: EncoderWithAction, q_value_offset: float = 0.0):
        super().__init__(encoder=encoder)

        # initial q_values for approximation
        self._q_value_offset = q_value_offset

        # get a new instance or clone a frozen copy
        self._fc0 = type(self._fc)(encoder.get_feature_size(), 1)

        # copy weights and stuff
        self._fc0.load_state_dict(self._fc.state_dict())

        # freeze model by freezing parameters
        for param in self._fc0.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x, action)) - self._fc0(self._encoder(x, action)) + self._q_value_offset)
