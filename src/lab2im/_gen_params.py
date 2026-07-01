import math
from numbers import Number
import typing as ty
from dataclasses import dataclass

UniformBounds: ty.TypeAlias = tuple[Number, Number]
_ShapeLike: ty.TypeAlias = int | ty.Sequence[int]

# these are all SynthSeg's defaults, taken from my personal fork (https://www.github.com/vcasellesb/SynthSeg)
@dataclass
class GenerationParams:
    # Spatial parameters
    ## Affine:
    rotation_bounds: UniformBounds = (-15 * math.pi / 180, 15 * math.pi / 180)
    scaling_bounds: UniformBounds = (0.8, 1.2)
    shearing_bounds: UniformBounds = (-0.02, 0.02)
    translation_bounds: UniformBounds = (0, 0) # I would like to change this
    ## Nonlin:
    svf_std_bounds: UniformBounds = (0, 4.)
    small_svf_size: _ShapeLike | None = None
    small_svf_factor: float | ty.Sequence[float] | None = 0.04
    sss_num_steps: int = 6

    # Flip parameters
    flip_axes: str | tuple[str] = 'R' # only right/left flipping
    p_per_axis: float = 0.5
    labels: ty.Sequence[int] = None
    right_labels: ty.Sequence[int] = None
    left_labels: ty.Sequence[int] = None

    # GMM parameters
    # labels = None (THIS IS SHARED WITH FLIP)
    means: ty.Sequence[Number] = None
    stds: ty.Sequence[Number] = None
    num_channels: int = 1
    means_bounds: UniformBounds = (0, 250)
    stds_bounds: UniformBounds = (0, 30)

    # Bias field parameters
    bias_field_std_bounds: UniformBounds = (0, 0.7)
    small_bias_field_shape: _ShapeLike = None
    small_bias_field_factor: float | ty.Sequence[float] = 0.025
    same_bias_field_per_channel: bool = False

    # Gaussian Noise
    noise_std_bounds: UniformBounds = (0, 0.1)
    noise_equal_for_all_channels: bool = False
    p_noise: float = 0.95

    normalization_method: ty.Literal['zscore', 'nonorm', 'minmax'] = 'minmax'

    gamma_std: float = 0.5

    # Low resolution simulation
    isotropic_upper_bounds: float = 4.
    anisotropic_upper_bounds: float = 8.
    aniso_dim: int | None = None
    p_input_resolution: float = 0.05 # These four are where I deviate from original SynthSeg...
    prob_isotropic: float = 0.25
    prob_anisotropic: float = 0.5
    prob_fully_random: float = 0.2
