from typing import Any, ClassVar, Dict

from d3rlpy.models.q_functions import QFunctionFactory, register_q_func_factory
from d3rlpy.models.torch import Encoder, EncoderWithAction
from d3rlpy_addons.models.torch.q_functions import (
    ContinuousDMeanQFunction,
    ContinuousDQRQFunction,
    DiscreteDMeanQFunction,
    DiscreteDQRQFunction,
)


class DMeanQFunctionFactory(QFunctionFactory):
    """Differential Regression Mean Q function factory class.
           It is a little trick to set all q_values to an initial fixed value so that
           optimistic, pessimistic or arbitrary q_value initialization can be set.

        References:
            * unpublished: Jose Antonio Martin H.

        Args:
            q_value_offset: the initial q_values for the nn approximation

        """

    TYPE: ClassVar[str] = "dmean"
    _q_value_offset: float

    def __init__(
        self,
        bootstrap: bool = False,
        share_encoder: bool = False,
        q_value_offset: float = 0,
    ):
        super().__init__(bootstrap, share_encoder)
        self._q_value_offset = q_value_offset

    def create_discrete(
        self, encoder: Encoder, action_size: int,
    ) -> DiscreteDMeanQFunction:
        return DiscreteDMeanQFunction(
            encoder, action_size, self._q_value_offset
        )

    def create_continuous(
        self, encoder: EncoderWithAction,
    ) -> ContinuousDMeanQFunction:
        return ContinuousDMeanQFunction(encoder, self._q_value_offset)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "bootstrap": self._bootstrap,
            "share_encoder": self._share_encoder,
        }


class DQRQFunctionFactory(QFunctionFactory):
    """Differential Quantile Regression Q function factory class.
       It is a little trick to set all q_values to an initial fixed value so that
       optimistic, pessimistic or arbitrary q_value initialization can be set.

    References:
        * unpublished: Jose Antonio Martin H.

    Args:
        n_quantiles: the number of quantiles.
        q_value_offset: the initial q_values for the nn approximation

    """

    TYPE: ClassVar[str] = "dqr"
    _n_quantiles: int
    _q_value_offset: float

    def __init__(
        self,
        bootstrap=False,
        share_encoder=False,
        n_quantiles: int = 32,
        q_value_offset: float = 0,
    ):
        super().__init__(bootstrap, share_encoder)
        self._n_quantiles = n_quantiles
        self._q_value_offset = q_value_offset

    def create_discrete(
        self, encoder: Encoder, action_size: int
    ) -> DiscreteDQRQFunction:
        return DiscreteDQRQFunction(
            encoder, action_size, self._n_quantiles, self._q_value_offset
        )

    def create_continuous(
        self, encoder: EncoderWithAction,
    ) -> ContinuousDQRQFunction:
        return ContinuousDQRQFunction(
            encoder, self._n_quantiles, self._q_value_offset
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "n_quantiles": self._n_quantiles,
            "q_value_offset": self._q_value_offset,
        }

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    @property
    def q_value_offset(self) -> float:
        return self._q_value_offset

    @property
    def share_encoder(self) -> bool:
        return False

    @property
    def bootstrap(self) -> bool:
        return False


register_q_func_factory(DQRQFunctionFactory)
register_q_func_factory(DMeanQFunctionFactory)
