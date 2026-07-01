import torch

from .base import BaseTransform


class GammaTransform(BaseTransform):

    def __init__(self,
                 gamma_std,
                 ):
        """
        ## WARNING
        Assumes that the step prior to this transform has normalized `image` to **strictly positive** values!
        Otherwise NaNs will be generated!
        """

        self.gamma_std = gamma_std
        self.epsilon = 1e-7


    def get_parameters(self, data):
        img = data.get(self.image_key)
        gamma_size = (img.shape[0], *((1,) * (img.ndim-1)))
        gammas = torch.randn(size=gamma_size, device=img.device) * self.gamma_std
        gammas.exp_()
        return {'gammas': gammas}


    def apply(self, data, **params):
        img: torch.Tensor = data[self.image_key]
        img.pow_(params['gammas'])
        data[self.image_key] = img
        return data


if __name__ == "__main__":
    gamma = GammaTransform(2)

    data = torch.randn((2, 128, 128, 128))
    print(gamma(**{'im': data})['im'].shape)
    gamma = GammaTransform(2)
    print(gamma(**{'im': data})['im'].shape)

    gamma = GammaTransform(2)
    # Use rand (strictly [0, 1]) instead of randn
    data = torch.rand((2, 128, 128, 128)) 

    out = gamma(**{'im': data})['im']
    print(out.shape)

    # Quick sanity check that bounds are preserved
    print(f"Min: {out.min().item()}, Max: {out.max().item()}")