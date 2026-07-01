import typing as ty
import torch

from .base import BaseTransform



class NormalizationTransform(BaseTransform):

    def __init__(self,
                 method: ty.Literal["zscore", "nonorm", "minmax"] | None
                 ):

        self.epsilon = torch.tensor([1e-8])
        mapping = {
            'zscore': self._zscore_norm,
            'minmax': self._minmax_norm,
            'nonorm': self._no_norm,
            None: self._no_norm
        }

        self.fn = mapping[method.lower()]

    def _no_norm(self, data, axis):
        return data

    def _zscore_norm(self, data: torch.Tensor, axis: tuple[int]):
        mean = data.mean(dim=axis, keepdim=True)
        std = data.std(dim=axis, keepdim=True)
        denom = torch.maximum(std, self.epsilon.to(data.device))
        return (data - mean) / denom

    def _minmax_norm(self, data: torch.Tensor, axis: tuple[int]):
        _min = data.amin(dim=axis, keepdim=True)
        _max = data.amax(dim=axis, keepdim=True)
        denom = torch.maximum(_max - _min, self.epsilon.to(data.device))
        return (data - _min) / denom

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

    data = torch.randn((2, 128, 128, 128))

    transf2 = NormalizationTransform(None)
    assert (transf(**{'im': data})['im'].shape) == (transf2(**{'im': data})['im'].shape)