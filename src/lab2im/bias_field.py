import torch
import torch.nn.functional as F

from .sample import sample_uniform
from .base import BaseTransform
from ._small_shape import get_small_shape



class BiasFieldTransform(BaseTransform):

    def __init__(self,
                 std_bounds: tuple,
                 small_bias_field_shape,
                 small_bias_field_factor: float | tuple[float, ...] | None,
                 same_bias_field_per_channel: bool
                 ):

        super().__init__()

        self.std_bounds = std_bounds

        if small_bias_field_shape is None and small_bias_field_factor is None:
            msg = (
                'Both "small_bias_field_shape" and "small_bias_field_factor" cannot be None.'
            )
            raise ValueError(msg)

        self.small_bias_field_shape = small_bias_field_shape
        self.small_bias_field_factor = small_bias_field_factor

        self.same_bias_field_per_channel = same_bias_field_per_channel


    def _get_small_shape(self, im: torch.Tensor) -> list[int]:
        tensor_shape = list(im.shape)
        if self.same_bias_field_per_channel:
            # artificially make all channels share same bias field
            tensor_shape[0] = 1
        return get_small_shape(tensor_shape, self.small_bias_field_shape, self.small_bias_field_factor)


    def _sample_std(self, im: torch.Tensor):
        # we cannot simply sample a scalar as if we need to sample nchannel stds then when multiplying
        # with the random normal sample in get_parameters it will break due to different dimensions
        std_shape = [1,] * im.ndim
        if not self.same_bias_field_per_channel:
            std_shape[0] = im.shape[0]
        return sample_uniform(std_shape, *self.std_bounds, dtype=im.dtype, device=im.device)


    def get_parameters(self, data: dict):
        image = data.get(self.image_key)
        std = self._sample_std(image)
        small_bias_shape = self._get_small_shape(image)
        small_bias_field = torch.randn(small_bias_shape, dtype=image.dtype, device=image.device) * std

        bias_field_full_shape = F.interpolate(
            small_bias_field[None],
            size=image.shape[1:],
            mode='trilinear',
            align_corners=False
        )[0]
        return {'bias_field': bias_field_full_shape}

    def apply(self, data, **params):
        bias_field: torch.Tensor = params.get('bias_field')
        bias_field.exp_()
        data[self.image_key] *= bias_field
        return data


if __name__ == "__main__":
    bfc = BiasFieldTransform(std_bounds=(0, 20),
                             small_bias_field_factor=0.1,
                             small_bias_field_shape=None,
                             same_bias_field_per_channel=False)

    data = torch.randn(size=(2, 128, 128, 128))
    assert (bfc(data).shape) == data.shape

    bfc = BiasFieldTransform(std_bounds=(0, 20),
                             small_bias_field_factor=0.1,
                             small_bias_field_shape=(13, 13, 13),
                             same_bias_field_per_channel=False)

    data = torch.randn(size=(2, 128, 128, 128))
    assert (bfc(data).shape) == data.shape

    bfc = BiasFieldTransform(std_bounds=(0, 20),
                             small_bias_field_factor=0.1,
                             small_bias_field_shape=None,
                             same_bias_field_per_channel=True)

    data = torch.randn(size=(2, 128, 128, 128))
    assert (bfc(data).shape) == data.shape