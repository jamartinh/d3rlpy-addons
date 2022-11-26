from typing import List, Optional, Sequence

import torch
from torch import nn

from d3rlpy.models.torch.encoders import _PixelEncoder, PixelEncoder, PixelEncoderWithAction


class MyConvEncoder(_PixelEncoder):

    def __init__(self, observation_shape: Sequence[int],
                 filters: Optional[List[Sequence[int]]] = None,
                 feature_size: int = 512,
                 use_batch_norm: bool = False,
                 dropout_rate: Optional[float] = False,
                 activation: nn.Module = nn.ReLU()
                 ):
        _PixelEncoder.__init__(self,
                               observation_shape=tuple([1] + list(observation_shape)),
                               filters=filters,
                               feature_size=feature_size,
                               use_batch_norm=use_batch_norm,
                               dropout_rate=dropout_rate,
                               activation=activation)


class ConvEncoder(MyConvEncoder, PixelEncoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = super().forward(torch.unsqueeze(x, 1))
        return h


class ConvEncoderWithAction(MyConvEncoder, PixelEncoderWithAction):
    def __init__(
            self,
            observation_shape: Sequence[int],
            action_size: int,
            filters: Optional[List[Sequence[int]]] = None,
            feature_size: int = 512,
            use_batch_norm: bool = False,
            dropout_rate: Optional[float] = None,
            discrete_action: bool = False,
            activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        MyConvEncoder.__init__(self,
                               observation_shape=observation_shape,
                               filters=filters,
                               feature_size=feature_size,
                               use_batch_norm=use_batch_norm,
                               dropout_rate=dropout_rate,
                               activation=activation,
                               )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = super().forward(torch.unsqueeze(x, 1), action)
        return h
