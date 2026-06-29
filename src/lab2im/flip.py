import typing as ty
import warnings
import torch

from .base import BaseTransform


class FlipTransform(BaseTransform):


    def __init__(self,
                 flip_axes: str | tuple[str, ...],
                 p_per_axis: float,
                 labels: ty.Sequence[int] | None,
                 right_labels: ty.Sequence[int] | None,
                 left_labels: ty.Sequence[int] | None
                 ):

        if isinstance(flip_axes, str):
            flip_axes = (flip_axes, )

        # I am so fucking stupid...
        axis_map = {'L': 'R', 'R': 'R', 'P': 'A', 'A': 'A', 'I': 'S', 'S': 'S'}
        allowed_axes_values = axis_map.keys()
        if any(ax not in allowed_axes_values for ax in flip_axes):
            msg = (
                f'Got unrecognized flip_axes values. The allowed values are: {" ".join(allowed_axes_values)}.'
                f'\nGot the following values: {" ".join(flip_axes)}'
            )
            raise ValueError(msg)

        self.flip_axes = flip_axes
        self.axis_map = axis_map

        self.p_per_axis = p_per_axis

        self.labels = torch.as_tensor(labels or [], dtype=torch.long)
        self.right_labels = torch.as_tensor(right_labels or [], dtype=torch.long)
        self.left_labels = torch.as_tensor(left_labels or [], dtype=torch.long)

        if self.right_labels.numel() != self.left_labels.numel():
            msg = 'If left_labels and right_labels are provided, they have to be of the same length!'
            raise ValueError(msg)


    def get_max_label(self,
                      class_vals: torch.Tensor
                      ):
        return max(
            class_vals.max().item(),
            self.right_labels.max().item() if len(self.right_labels) else 0,
            self.left_labels.max().item() if len(self.left_labels) else 0,
            self.labels.max().item() if len(self.labels) else 0
        )


    def get_parameters(self, data: dict):
        seg = data[self.segmentation_key]

        # by default assume that the segmentation is in RAS. This is not good, I know.
        ax_codes = data.get('axis_codes')
        if ax_codes is None:
            warnings.warn('axis_codes not provided. Assuming input data is in RAS orientation.')
            ax_codes = tuple('RAS')

        candidate_axes = [ax_codes.index(self.axis_map[v]) for v in self.flip_axes]

        flip = torch.rand(len(candidate_axes)) < self.p_per_axis
        if not torch.any(flip):
            return {'flip_axes': torch.tensor([]), 'lut': False}

        # if we do two flips, then we are essentially doing a 180 rotation, therefore
        # we no need to swap labels
        lut = None
        swap_labels = torch.sum(flip).item() % 2 != 0
        if swap_labels:
            # we construct lut
            seg_vals = torch.unique(seg.ravel())
            max_label = self.get_max_label(seg_vals)
            lut = torch.arange(0, max_label+1, dtype=torch.long, device=seg.device)
            lut[self.right_labels] = self.left_labels
            lut[self.left_labels] = self.right_labels

        flip_axes = torch.as_tensor([candidate_axes[i] for i in range(len(candidate_axes)) if flip[i]], dtype=torch.long)
        return {'flip_axes': flip_axes, 'lut': lut, 'swap_labels': swap_labels}


    def apply(self, data, **params):
        flip_axes: torch.Tensor = params.get('flip_axes')
        if len(flip_axes) == 0:
            return data
        seg = data[self.segmentation_key].clone().long()
        if params.get('swap_labels'):
            lut = params.get('lut')
            seg = lut[seg]

        flip_axes = (flip_axes + 1).long().tolist()
        data[self.segmentation_key] = torch.flip(seg, flip_axes)
        return data


if __name__ == "__main__":
    # flip = FlipTransform(('R', 'A', 'S'))
    # try:
    #     flip = FlipTransform(('D', 'U', 'M', 'M', 'Y'))
    # except ValueError as e:
    #     print(e.args[0])
    pass