import typing as ty
import random
import warnings
import torch
import torch.nn.functional as F

from . import (
    sample,
    base,
    fft_conv as fft
)


def compute_kernel_size(sigma, truncate):
    return round_to_nearest_odd(sigma * truncate + 0.5)

def build_gaussian_kernel(sigma: float, truncate: float = 4.) -> torch.Tensor:
    kernel_size = compute_kernel_size(sigma, truncate=truncate)
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d

def round_to_nearest_odd(n):
    rounded = round(n)
    # If the rounded number is odd, return it
    if rounded % 2 == 1:
        return rounded
    # If the rounded number is even, adjust to the nearest odd number
    return rounded + 1 if n - rounded >= 0 else rounded - 1


def blur_dimension(img: torch.Tensor, sigma: float, dim_to_blur: int,
                   spatial_dims: int = None, force_use_fft: bool = None, truncate: float = 6
                   ):
    """
    Smoothes an input image with a 1D Gaussian kernel along the specified dimension.
    The function supports 1D, 2D, and 3D images.

    :param img: Input image tensor with shape (C, X), (C, X, Y), or (C, X, Y, Z),
                where C is the channel dimension and X, Y, Z are spatial dimensions.
    :param sigma: The standard deviation of the Gaussian kernel.
    :param dim_to_blur: The dimension along which to apply the Gaussian blur (0 for X, 1 for Y, 2 for Z).
    :return: The blurred image tensor.
    """
    assert img.ndim - 1 > dim_to_blur, "dim_to_blur must be a valid spatial dimension of the input image."
    # Adjustments for kernel based on image dimensions
    spatial_dims = spatial_dims or (img.ndim - 1)
    kernel = build_gaussian_kernel(sigma, truncate=truncate)

    ksize = kernel.numel()

    # Dynamically set up padding, convolution operation, and kernel shape based on the number of spatial dimensions
    if not force_use_fft:
        conv_op = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial_dims]
    else:
        conv_op = fft.fft_conv

    # Adjust kernel and padding for the specified blur dimension and input dimensions
    ksize = kernel.shape[0]
    pad = ksize // 2

    # Adjust kernel and padding for the specified blur dimension and input dimensions
    # I almost suffered a stroke trying to think of the following 7 lines
    expand_dims = [1] * (spatial_dims + 2)
    expand_dims[dim_to_blur + 2] = ksize
    kernel = kernel.view(expand_dims)

    padding = [0] * (spatial_dims * 2)
    padding_slicer = slice(dim_to_blur * 2, dim_to_blur * 2 + 2)
    padding[padding_slicer] = [pad, pad]

    # Apply padding
    img_padded = F.pad(img, padding[::-1], mode="reflect")

    # Apply convolution
    # remember that weights are [c_out, c_in, ...]
    img_blurred = conv_op(
        img_padded[None],
        kernel.expand(img_padded.shape[0], *[-1] * (kernel.ndim - 1)),
        groups=img_padded.shape[0]
    )[0]
    return img_blurred


class LowResolutionTransform(base.BaseTransform):

    gaussian_blur_sigma_coeff = 0.75

    def __init__(self,
                 isotropic_upper_bounds: float,
                 anisotropic_upper_bounds: float,
                 aniso_dim: int | None,
                 prob_input_resolution: float,
                 prob_isotropic: float,
                 prob_anisotropic: float,
                 prob_fully_random: float
                 ):

        super().__init__()

        self.iso_ubds = isotropic_upper_bounds
        self.aniso_ubds = anisotropic_upper_bounds

        self.aniso_dim = aniso_dim

        self.modes = ['input', 'iso', 'aniso', 'random']
        self.weights = [prob_input_resolution, prob_isotropic, prob_anisotropic, prob_fully_random]

        self.interpolate_mode = {
            2: 'bilinear',
            3: 'trilinear'
        }

        self.eps = 1e-4
        if abs(sum(self.weights) - 1) > self.eps:
            warnings.warn(f'The input probabilities provided don\'t sum to 1 within {self.eps} error.'
                          ' If you care at all about your credibility as a scientist, you\'ll change that.')


    def sample_resolution(self, input_resolution: torch.Tensor) -> torch.Tensor:
        mode = random.choices(self.modes, weights=self.weights)[0]
        if mode == "input":
            return input_resolution.clone()

        ndim = len(input_resolution)

        if mode == "iso":
            # we do torch.max as we cannot/do not want to simulate a higher resolution
            # than the lower resolution, i.e. if input_resolution is [1., 1., 3.]
            # we cannot simulate a isotropic resolution where the last dimension is < 3. mm
            lbds = torch.max(input_resolution).item()
            ubds = max(lbds, self.iso_ubds)
            return sample.sample_uniform(1, lbds, ubds).repeat(ndim)

        if mode == "aniso":
            aniso_dim = self.aniso_dim if self.aniso_dim is not None else random.randint(a=0, b=ndim-1)
            aniso_resolution = input_resolution.clone()
            lbds = aniso_resolution[aniso_dim].item()
            ubds = max(lbds, self.aniso_ubds)
            aniso_resolution[aniso_dim] = sample.sample_uniform(1, lbds, ubds).item()
            return aniso_resolution

        lbds = input_resolution.clone()
        ubds = self.aniso_ubds
        if not isinstance(ubds, ty.Iterable):
            ubds = [ubds] * ndim
        # do not let ubds be smaller than lbds, else we crash
        ubds = torch.maximum(torch.as_tensor(ubds), lbds)
        random_resolution = sample.sample_uniform(ndim, lbds, ubds)

        return random_resolution

    @staticmethod
    def get_gaussian_blurring_sigma(input_resolution: torch.Tensor, acquisition_resolution: torch.Tensor, thickness: torch.Tensor):
        coeff = LowResolutionTransform.gaussian_blur_sigma_coeff
        tissue_to_average = torch.minimum(acquisition_resolution, thickness)
        sigma = coeff * tissue_to_average / input_resolution
        sigma[tissue_to_average == input_resolution] = 0.5
        return sigma

    def get_parameters(self, data):
        image = data.get(self.image_key)
        spatial_dims = image.ndim-1
        input_resolution = torch.as_tensor(data.get('input_resolution', torch.ones(spatial_dims)))

        nchannels = image.shape[0]
        sigmas, acquisition_resolutions = [], []
        for _ in range(nchannels):
            acquisition_resolution = self.sample_resolution(input_resolution)
            thickness = sample.sample_uniform(spatial_dims, input_resolution, acquisition_resolution)
            gaussian_blurring_sigma = self.get_gaussian_blurring_sigma(input_resolution, acquisition_resolution, thickness)
            sigmas.append(gaussian_blurring_sigma)
            acquisition_resolutions.append(acquisition_resolution)
        return {'sigmas': sigmas, 'acquisition_resolutions': acquisition_resolutions}

    def apply(self, data, **params: list):
        image = data.get(self.image_key)
        _, *spatial_shape = image.shape
        spatial_dims = len(spatial_shape)

        input_resolution = data.get('input_resolution', torch.ones(spatial_dims))
        input_resolution = torch.as_tensor(input_resolution)
        spatial_shape_as_tensor = torch.as_tensor(spatial_shape)

        out = torch.zeros_like(image)
        for c, (gaussian_sigmas_per_dim, acquisition_resolution) in enumerate(zip(
            params.get('sigmas'), params.get('acquisition_resolutions')
        )):
            channel_data = image[c].clone().unsqueeze(0)
            for d in range(spatial_dims):
                sigma = gaussian_sigmas_per_dim[d].item()
                if sigma > self.eps:
                    channel_data = blur_dimension(channel_data, sigma, d, spatial_dims=spatial_dims)

            down_shape = (spatial_shape_as_tensor * input_resolution / acquisition_resolution).long()
            down_shape = torch.maximum(down_shape, torch.ones_like(down_shape)).tolist()
            channel_data = F.interpolate(
                channel_data[None], size=down_shape, mode='nearest'
            )
            out[c] = F.interpolate(
                channel_data, size=spatial_shape, mode=self.interpolate_mode[spatial_dims], align_corners=False
            )[0, 0]

        data[self.image_key] = out
        return data



if __name__ == "__main__":
    input_shape = (2, 128, 140, 120)
    data = torch.randn(input_shape)
    LowRes = LowResolutionTransform(3., 5., None, 0.25, 0.25, 0.30, 0.25)
    out = LowRes(im=data, input_resolution=[1., 1., 1.])
    assert list(out.shape[1:]) == list(input_shape[1:])