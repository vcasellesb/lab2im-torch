import typing as ty
from numbers import Number
import torch
import torch.nn.functional as F

from .base import BaseTransform
from .sample import sample_uniform
from ._small_shape import get_small_shape


def create_affine_matrix_2d():
    pass

def create_rotation_matrix_3d(rotation_angles: torch.Tensor) -> torch.Tensor:
    rotx, roty, rotz = rotation_angles
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(rotx), -torch.sin(rotx)],
                       [0, torch.sin(rotx), torch.cos(rotx)]])
    Ry = torch.tensor([[torch.cos(roty), 0, torch.sin(roty)],
                       [0, 1, 0],
                       [-torch.sin(roty), 0, torch.cos(roty)]])
    Rz = torch.tensor([[torch.cos(rotz), -torch.sin(rotz), 0],
                       [torch.sin(rotz), torch.cos(rotz), 0],
                       [0, 0, 1]])
    return Rz @ Ry @ Rx

def create_shearing_matrix_3d(shearing_factors: torch.Tensor) -> torch.Tensor:

    shearing_matrix = torch.ones((3, 3), dtype=shearing_factors.dtype)
    # I follow Synthseg's specification, which maximizes randomness and not so much mathematical convention
    # I am transversing the 3x3 affine matrix row-first (omitting 1s, of course)
    (
        kxy,
        kxz,
        kyx,
        kyz,
        kzx,
        kzy
    ) = shearing_factors
    shearing_matrix[0, 1] = kxy
    shearing_matrix[0, 2] = kxz
    shearing_matrix[1, 0] = kyx
    shearing_matrix[1, 2] = kyz
    shearing_matrix[2, 0] = kzx
    shearing_matrix[2, 1] = kzy
    return shearing_matrix


def create_affine_matrix_3d(rotation_angles, scaling_factors, shearing_factors):
    rotation_matrix = create_rotation_matrix_3d(rotation_angles)
    scaling_matrix = torch.diag(scaling_factors).to(rotation_matrix.dtype)
    shearing_matrix = create_shearing_matrix_3d(shearing_factors)
    A = shearing_matrix @ rotation_matrix @ scaling_matrix
    return A


def create_identity_grid(spatial_shape: tuple | torch.Tensor, indexing: ty.Literal['ij', 'xy']):
    spatial_shape = tuple(spatial_shape)
    half_size = [(spatial_shape[i] - 1) / 2 for i in range(len(spatial_shape))]
    space = [torch.linspace(-half_size[i], half_size[i], steps=spatial_shape[i]) for i in range(len(spatial_shape))]
    grid = torch.meshgrid(*space, indexing=indexing)
    return torch.stack(grid, dim=-1)

def convert_grid_to_grid_for_grid_sample(grid: torch.Tensor, original_shape, indexing: ty.Literal['ij', 'xy']):
    """
    :param indexing: whether the grid was constructed using `ij` or `xy` indexing. `torch.functional.grid_sample`
    requires `xy` indexing, so there's that.
    """
    original_shape = torch.tensor(original_shape, dtype=grid.dtype, device=grid.device)
    grid = grid / (original_shape / 2)
    if indexing == "ij":
        grid = torch.flip(grid, (grid.ndim - 1,))
    return grid

def get_permutation_mappings(spatial_shape) -> tuple[tuple, tuple]:
    """
    Given a spatial shape of [H W D], the deformation grid shape will be [H W D 3].

    This function provides the tuples to be passed on to torch.Tensor.permute to map from [H W D 3] to [3 H W D],
    as required by `grid_sample` or `interpolate`.
    """
    ndim = len(spatial_shape)
    fwd_map = (ndim, *list(range(ndim)))
    bwd_map = (*list(range(1, ndim+1)), 0)
    return fwd_map, bwd_map


def sss(svf: torch.Tensor, num_steps: int,
        mode: str, padding_mode: str,
        identity_grid: torch.Tensor,
        indexing: str = 'ij'
        ):
    """
    :param svf: input stationary velocity field of shape [*`spatial_dims`, `ndims`]
    """
    # some sexy songs 4 u
    # Scale and square algorithm
    # we compose the num_step times downscaled (in magnitude) SVF with itself num_step times
    # to approximate its integration or exponential map
    *spatial_shape, ndims = svf.shape
    if ndims != len(spatial_shape):
        msg = (
            'Please provide the svf with shape [SPATIAL_DIM_1, SPATIAL_DIM_2, ..., SPATIAL_DIM_N, N].\n'
            f'Got a tensor of shape: {spatial_shape + [ndims]}'
        )
        raise ValueError(msg)

    # we map identity grid to [-1, 1] as grid_sample wants
    identity_grid = convert_grid_to_grid_for_grid_sample(identity_grid, spatial_shape, indexing)

    # F.grid_sample requires scaled/input to be [N, C, *(spatial dims)]
    # while the flow field grid to be: [N, *(spatial dims), Ndims(len(spatial dims))]
    # these will allow us to bring num channels/dimensions to front and back of tensor
    permute_mapping_fwd, permute_mapping_bwd = get_permutation_mappings(spatial_shape)

    # start the party.
    # Let's say num_steps is 6. We scale SVF by dividing it by 2**6 to approximate its integration with a Taylor Series
    # we do: step 1 - exp(v/64) o exp(v/64) = exp(v/32)
    #        step 2 - exp(v/32) o exp(v/32) = exp(v/16)
    #            ...
    # until we end up getting something very close to exp(v) = u (displacement field, or offsets)
    scaled = svf / (2 ** num_steps)
    for _ in range(num_steps):
        # identity_grid is already converted to [-1, 1], now we have to convert our SVF/displacement field
        # to [-1, 1]
        for_grid_sample = convert_grid_to_grid_for_grid_sample(scaled, spatial_shape, indexing=indexing)

        scaled = scaled.permute(permute_mapping_fwd)
        scaled += F.grid_sample(
            scaled[None],
            (identity_grid + for_grid_sample)[None],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False
        )[0]
        scaled = scaled.permute(permute_mapping_bwd)
    return scaled


class SpatialTransform(BaseTransform):

    def __init__(self,
                 rotation_bounds: tuple[Number, Number],
                 scaling_bounds: tuple[Number, Number],
                 shearing_bounds: tuple[Number, Number],
                 translation_bounds: tuple[Number, Number],
                 svf_std_bounds: tuple[Number, Number],
                 small_svf_size: Number | ty.Iterable[Number] | None,
                 small_svf_factor: float | ty.Sequence[float] | None,
                 sss_num_steps: int
                 ):
        """
        :param rotation_bounds: has to be in radians
        """

        super().__init__()

        self.rotation_bounds = rotation_bounds
        self.scaling_bounds = scaling_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds

        self.svf_std_bounds = svf_std_bounds

        self.small_svf_size = small_svf_size
        self.small_svf_factor = small_svf_factor

        self.sss_num_steps = sss_num_steps

        self._cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._grid_indexing: ty.Literal['ij', 'xy'] = 'ij'

        self.mapping_zoom = {2: 'bilinear',
                             3: 'trilinear'}


    def _get_identity_grid(self, shape: tuple | torch.Size):
        """
        Returns grid that is centered around `shape` center.
        """
        key = tuple(shape)
        try:
            grid = self._cache[key]
        except KeyError:
            grid = create_identity_grid(shape, self._grid_indexing)
            self._cache[key] = grid
        # important! return copy, as we don't want to mess up the cache
        return grid.clone()


    def sample_random_deformation_field(self, spatial_shape, device, dtype) -> torch.Tensor:
        ndim = len(spatial_shape)
        std = sample_uniform(1, *self.svf_std_bounds, dtype=dtype, device=device)
        small_svf_size = get_small_shape([ndim, *spatial_shape], self.small_svf_size, self.small_svf_factor)
        small_svf = (
            torch.randn(small_svf_size, dtype=dtype, device=device) * std
        )
        # def zoom(data: torch.Tensor, size, mode: str = 'trilinear'):
        #     *spatial_shape, ndim = data.shape
        #     permute_fwd, permute_bwd = get_permutation_mappings(spatial_shape)
        #     # bring num dimensions/channels to front of tensor and make 5D to appease torch's interpolate
        #     data = data.permute(permute_fwd)[None]
        #     data = F.interpolate(data, size=size, mode=mode, align_corners=False)[0]
        #     return data.permute(permute_bwd)
        # Before I intiialized small_svf as [x, y, z, ndims]
        # now initialize it directly as [ndims, x, y, z] (as interpolate requires) and then bring it to what grid_sample requires
        svf_full_size = F.interpolate(small_svf[None], size=spatial_shape, mode=self.mapping_zoom[ndim], align_corners=False)[0]
        svf_full_size = svf_full_size.permute(*list(range(1, ndim+1)), 0)

        # from Gemini: use trilinear interpolation and border padding mode for interpolating
        # the deformation field. padding = border essentially means that points outside of original
        # grid are given the value of the border/edge
        # WHYY WHY THE FUCK DOES GRID_SAMPLE REQUIRE BILINEAR INTERPOLATION WHEN INTERPOLATING
        # A 3D IMAGE??
        identity_grid = self._get_identity_grid(spatial_shape).to(device=device, dtype=dtype)
        offsets = sss(svf_full_size, self.sss_num_steps,
                      mode=self.mapping_zoom[ndim-1], padding_mode='border',
                      identity_grid=identity_grid, indexing=self._grid_indexing)
        return offsets

    def get_affine_parameters(self, ndim: int) -> dict:
        rotation_angles = sample_uniform(ndim, *self.rotation_bounds)
        shearing_factors = sample_uniform((ndim * (ndim-1)), *self.shearing_bounds)
        scaling_factors = sample_uniform(ndim, *self.scaling_bounds)
        translation = sample_uniform(ndim, *self.translation_bounds)

        affine_matrix_params = {
            'rotation_angles': rotation_angles,
            'shearing_factors': shearing_factors,
            'scaling_factors': scaling_factors
        }
        return affine_matrix_params, translation

    def create_affine_matrix(self, spatial_shape) -> tuple[torch.Tensor, torch.Tensor]:
        ndim = len(spatial_shape)
        affine_params, translation = self.get_affine_parameters(ndim)
        if ndim == 3:
            return create_affine_matrix_3d(**affine_params), translation
        return create_affine_matrix_2d(**affine_params), translation

    def get_parameters(self, data):
        seg = data.get(self.segmentation_key)
        spatial_shape = seg.shape[1:]
        device = seg.device
        # the following code introduced a bug:
        # dtype = seg.dtype
        # do not generate a SVF inheriting dtype from a segmentation, dummy
        dtype = torch.float32

        identity_grid = self._get_identity_grid(spatial_shape).to(device=device, dtype=dtype)

        offsets = self.sample_random_deformation_field(spatial_shape, device, dtype)
        # we subtract the mean displacement field so that its mean is 0
        # since we model translations separately this should be more correct - source: Gemini
        offsets -= offsets.mean(dim=tuple(range(len(spatial_shape))), keepdim=True) # offsets is of dimension [X, Y, Z, 3]
        grid = identity_grid + offsets

        # now get affine transform and apply it to current grid
        affine_mat, translation_vector = self.create_affine_matrix(spatial_shape)
        translation_vector = translation_vector.to(device=device, dtype=dtype)
        # grid stores each voxel's vector as a row vector, [x, y,(, z)]
        # then we have to manipulate affine's dimensionality to take that into account
        affine_mat = affine_mat.T.to(device=device, dtype=dtype)
        grid = torch.matmul(grid, affine_mat) + translation_vector
        grid = convert_grid_to_grid_for_grid_sample(grid, original_shape=spatial_shape, indexing=self._grid_indexing)
        return {'grid': grid}

    def apply(self, data, **params):
        seg = data.get(self.segmentation_key).clone().float()
        assert seg.ndim > 3, (
            f'Please pass your input data as 4D dimensional tensors. Got shape: {tuple(data.shape)}'
        )
        grid = params.get('grid')
        # from Gemini as well, use nearest for interpolating the segmentation,
        # with out-of-bounds points being given a zero value
        seg = F.grid_sample(
            seg[None],
            grid[None],
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )[0]
        data[self.segmentation_key] = seg.long()
        return data


if __name__ == "__main__":
    import math
    import nibabel as nib
    import numpy as np

    def to_radians(degrees):
        return degrees * math.pi / 180

    spatial_transf = SpatialTransform(rotation_bounds=(to_radians(-15), to_radians(15)),
                                      scaling_bounds=(0.8, 1.2),
                                      shearing_bounds=(-0.02, 0.02),
                                      translation_bounds=(-5, 5),
                                      small_svf_size=(10, 10, 10),
                                      small_svf_factor=None,
                                      svf_std_bounds=(0, 2),
                                      sss_num_steps=6,
                                      interp_mode='nearest',
                                      padding_mode='zeros'
                                      )

    label = nib.load('/Users/vicentcaselles/work/research/project_MARCOS/Multiple-Sclerosis-TIMILS/subj5/flair_bfc_filled_NeuroMorph_Parcellation_cleaned.nii.gz')
    data = torch.from_numpy(np.asarray(label.dataobj))[None]
    transformed = spatial_transf(**{'seg': data})['seg'][0].numpy().astype(np.uint16)

    nib.save(nib.Nifti1Image(transformed, affine=label.affine), 'test_1.nii.gz')