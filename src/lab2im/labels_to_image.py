from . import (
    spatial,
    flip,
    gmm,
    bias_field as bf,
    noise,
    lowres
)

from ._gen_params import GenerationParams

from .base import RandomTransform
from .compose import ComposeTransforms


class LabelsToImage:

    def __init__(self, generation_parameters: GenerationParams):
        self.transforms = self.build_transforms(generation_parameters)
        self.generation_parameters = generation_parameters

    def build_transforms(self, generation_parameters: GenerationParams):
        transforms = []

        transforms.append(spatial.SpatialTransform(
            rotation_bounds=generation_parameters.rotation_bounds,
            scaling_bounds=generation_parameters.scaling_bounds,
            shearing_bounds=generation_parameters.shearing_bounds,
            translation_bounds=generation_parameters.translation_bounds,
            svf_std_bounds=generation_parameters.svf_std_bounds,
            small_svf_size=generation_parameters.small_svf_size,
            small_svf_factor=generation_parameters.small_svf_factor,
            sss_num_steps=generation_parameters.sss_num_steps
        ))

        transforms.append(flip.FlipTransform(
            flip_axes=generation_parameters.flip_axes,
            p_per_axis=generation_parameters.p_per_axis,
            labels=generation_parameters.labels,
            right_labels=generation_parameters.right_labels,
            left_labels=generation_parameters.left_labels
        ))

        transforms.append(gmm.SampleGMM(
            labels=generation_parameters.labels,
            means=generation_parameters.means,
            stds=generation_parameters.stds,
            num_channels=generation_parameters.num_channels,
            means_bounds=generation_parameters.means_bounds,
            stds_bounds=generation_parameters.stds_bounds
        ))

        transforms.append(bf.BiasFieldTransform(
            std_bounds=generation_parameters.bias_field_std_bounds,
            small_bias_field_shape=generation_parameters.small_bias_field_shape,
            small_bias_field_factor=generation_parameters.small_bias_field_factor,
            same_bias_field_per_channel=generation_parameters.same_bias_field_per_channel
        ))

        transforms.append(RandomTransform(
            noise.NoiseAdditionTransform(
                noise_std_bounds=generation_parameters.noise_std_bounds,
                equal_for_all_channels=generation_parameters.noise_equal_for_all_channels
            ),
            p=generation_parameters.p_noise
        ))

        transforms.append(lowres.LowResolutionTransform(
            isotropic_upper_bounds=generation_parameters.isotropic_upper_bounds,
            anisotropic_upper_bounds=generation_parameters.anisotropic_upper_bounds,
            aniso_dim=generation_parameters.aniso_dim,
            prob_input_resolution=generation_parameters.p_input_resolution,
            prob_isotropic=generation_parameters.prob_isotropic,
            prob_anisotropic=generation_parameters.prob_anisotropic,
            prob_fully_random=generation_parameters.prob_fully_random
        ))

        return ComposeTransforms(transforms)


    def __call__(self, **data):
        return self.transforms.__call__(**data)



if __name__ == "__main__":
    import numpy as np
    import nibabel as nib
    import torch

    label = nib.load('/Users/vicentcaselles/work/research/project_MARCOS/Multiple-Sclerosis-TIMILS/subj5/flair_bfc_filled_NeuroMorph_Parcellation_cleaned.nii.gz')
    input_resolution = label.header.get_zooms()
    data = torch.from_numpy(np.asarray(label.dataobj))[None]
    lab2im = LabelsToImage(GenerationParams())
    data = lab2im(**{'seg': data, 'input_resolution': input_resolution})
    image = data.get('im')
    seg = data.get('seg')
    image = image.squeeze(0).numpy()
    seg = seg.squeeze(0).numpy().astype(np.uint16)

    new_data = nib.Nifti1Image(image, label.affine)
    new_seg = nib.Nifti1Image(seg, label.affine)
    new_data.to_filename('test_im.nii.gz')
    new_seg.to_filename('test_seg.nii.gz')

