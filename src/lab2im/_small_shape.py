import typing as ty
import torch


def get_small_shape(tensor_shape: ty.Sequence[int] | torch.Size,
                    small_shape_absolute: None | int | ty.Sequence[int] | torch.Tensor,
                    small_shape_factor: None | float | ty.Sequence[float] | torch.Tensor
                    ) -> list[int]:
    """
    Useful for generating small cute SVFs and then bringing them to big boy size.
    """
    channel_dim, *spatial_shape = tensor_shape

    if small_shape_absolute is not None:
        if isinstance(small_shape_absolute, int):
            return [channel_dim] + [small_shape_absolute] * len(spatial_shape)

        small_shape_absolute = list(small_shape_absolute)
        if len(small_shape_absolute) == len(spatial_shape):
            small_shape_absolute.insert(0, channel_dim)
        return small_shape_absolute

    if isinstance(small_shape_factor, float):
        small_shape_factor = (small_shape_factor,) * len(spatial_shape)

    spatial_shape = torch.as_tensor(spatial_shape)
    small_shape_factor = torch.as_tensor(small_shape_factor)
    small_bias_field_spatial_shape = torch.ceil(spatial_shape * small_shape_factor).long().tolist()
    return [channel_dim] + small_bias_field_spatial_shape


if __name__ == "__main__":
    tensor_shape = [2, 128, 128, 128]
    small_shape_absolute = [32, 32, 32]
    small_shape_factor = None

    small_shape1 = get_small_shape(tensor_shape, small_shape_absolute, small_shape_factor)
    small_shape2 = get_small_shape(tensor_shape, 32, 3)
    assert small_shape1 == small_shape2 == [2, 32, 32, 32]

    tensor_shape = [2, 128, 128, 128]
    small_shape_absolute = None
    small_shape_factor = [0.5, 0.5, 0.5]

    small_shape1 = get_small_shape(tensor_shape, small_shape_absolute, small_shape_factor)
    small_shape2 = get_small_shape(tensor_shape, small_shape_absolute, 0.5)
    assert small_shape1 == small_shape2 == [2, 64, 64, 64]
