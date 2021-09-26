from typing import Sequence

import torch

from d3rlpy.models.encoders import EncoderFactory
from .torch.policies import (
    EnsembleDeterministicPolicy,
    EnsembleCategoricalPolicy,
    EnsembleDeterministicResidualPolicy,
    EnsembleSquashedNormalPolicy,
)


def create_ensemble_deterministic_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    n_actors: int = 1,
) -> EnsembleDeterministicPolicy:
    encoder = encoder_factory.create(observation_shape)
    return EnsembleDeterministicPolicy(encoder, action_size, n_actors)


def create_deterministic_residual_policy(
    observation_shape: Sequence[int],
    action_size: int,
    scale: float,
    encoder_factory: EncoderFactory,
    n_actors: int = 1,
) -> EnsembleDeterministicResidualPolicy:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    return EnsembleDeterministicResidualPolicy(encoder, scale, n_actors)


def create_squashed_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> EnsembleSquashedNormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    return EnsembleSquashedNormalPolicy(
        encoder,
        action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )


def create_categorical_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> EnsembleCategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    return EnsembleCategoricalPolicy(encoder, action_size)
