from typing import Any, Dict, Optional, Sequence

import d3rlpy.models.torch.dynamics
from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.dynamics.base import DynamicsBase
from d3rlpy.dynamics.torch.probabilistic_ensemble_dynamics_impl import (
    ProbabilisticEnsembleDynamicsImpl, )
from d3rlpy.models.torch import ProbabilisticDynamicsModel

from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
from ..models.torch.dynamics import StateProbabilisticDynamicsModel


class MyProbabilisticEnsembleDynamics(ProbabilisticEnsembleDynamics):
    r"""MyProbabilistic ensemble dynamics.
    """
    def _create_impl(self, observation_shape: Sequence[int],
                     action_size: int) -> None:
        old_class = d3rlpy.dynamics.ProbabilisticEnsembleDynamics
        d3rlpy.models.torch.dynamics.ProbabilisticDynamicsModel = StateProbabilisticDynamicsModel
        self._impl = ProbabilisticEnsembleDynamicsImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            n_ensembles=self._n_ensembles,
            variance_type=self._variance_type,
            discrete_action=self._discrete_action,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()
        d3rlpy.dynamics.ProbabilisticEnsembleDynamics = old_class

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        return {"loss": loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.BOTH
