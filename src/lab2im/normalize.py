import typing as ty
import torch

from .base import BaseTransform



class NormalizationTransform(BaseTransform):

    def __init__(self,
                 method: ty.Literal["zscore"] | None
                 ):

        self.epsilon = torch.tensor([1e-8])
        mapping = {
            'zscore': self._zscore_norm
        }

        self.fn = mapping[method]

    def _zscore_norm(self, data: torch.Tensor, axis: tuple[int]):
        mean = data.mean(dim=axis, keepdim=True)
        std = data.std(dim=axis, keepdim=True)
        denom = torch.maximum(std, self.epsilon.to(data.device))
        return (data - mean) / denom

    def get_parameters(self, data):
        return {}

    def apply(self, data, **params):
        img = data[self.image_key]
        axis = tuple(range(1, img.ndim))
        data[self.image_key] = self.fn(img, axis)
        return data


if __name__ == "__main__":
    data = torch.randn((2, 128, 128, 128))

    transf = NormalizationTransform('zscore')
    print(transf(**{'im': data})['im'].shape)