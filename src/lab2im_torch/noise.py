import torch

from .sample import sample_uniform
from .base import BaseTransform


class NoiseAdditionTransform(BaseTransform):

    def __init__(self, noise_std_bounds, equal_for_all_channels: bool):
        self.noise_std_bounds = noise_std_bounds
        self.equal_for_all_channels = equal_for_all_channels

    def get_parameters(self, data):
        image = data.get(self.image_key)
        num_channels = max((not self.equal_for_all_channels) * image.shape[0], 1)
        std_shape = [num_channels] + [1] * (image.ndim-1)
        std = sample_uniform(std_shape, *self.noise_std_bounds, dtype=torch.float32, device=image.device)
        noise = torch.randn_like(image) * std
        return {'noise': noise}

    def apply(self, data, **params):
        noise = params.get('noise')
        data[self.image_key] += noise
        return data


if __name__ == "__main__":
    data = torch.randn(size=(2, 128, 128, 128))
    transf = NoiseAdditionTransform((0, 30), equal_for_all_channels=False)
    print(transf({'image': data}).shape)


    data = torch.randn(size=(2, 128, 128, 128))
    transf = NoiseAdditionTransform((0, 30), equal_for_all_channels=True)
    print(transf({'image': data}).shape)