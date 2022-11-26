from typing import ClassVar, Sequence

from d3rlpy.models.encoders import _create_activation, PixelEncoderFactory, register_encoder_factory
from d3rlpy_addons.models.torch.encoders import ConvEncoder, ConvEncoderWithAction


class ConvEncoderEncoderFactory(PixelEncoderFactory):
    """Pixel encoder factory class.

    This is the default encoder factory for image observation.

    Args:
        filters (list): list of tuples consisting with
            ``(filter_size, kernel_size, stride)``. If None,
            ``Nature DQN``-based architecture is used.
        feature_size (int): the last linear layer size.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "matrix-conv"

    def create(self, observation_shape: Sequence[int]) -> ConvEncoder:
        return ConvEncoder(
            observation_shape=observation_shape,
            filters=self._filters,
            feature_size=self._feature_size,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            activation=_create_activation(self._activation),
        )

    def create_with_action(
            self,
            observation_shape: Sequence[int],
            action_size: int,
            discrete_action: bool = False,
    ) -> ConvEncoderWithAction:
        return ConvEncoderWithAction(
            observation_shape=observation_shape,
            action_size=action_size,
            filters=self._filters,
            feature_size=self._feature_size,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            discrete_action=discrete_action,
            activation=_create_activation(self._activation),
        )


register_encoder_factory(ConvEncoderEncoderFactory)
