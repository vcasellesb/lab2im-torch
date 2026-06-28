import torch

from .sample import sample_uniform
from .base import BaseTransform


class SampleGMM(BaseTransform):

    def __init__(self,
                 labels,
                 means,
                 stds,
                 num_channels: int,
                 means_bounds,
                 stds_bounds
                 ):

        self.num_channels = num_channels
        self.means_bounds = means_bounds
        self.stds_bounds = stds_bounds


    def get_parameters(self, data):
        seg = data.get(self.segmentation_key)
        class_vals = torch.unique(seg.ravel(), sorted=True)
        max_label = int(class_vals[-1])
        means_lut = torch.zeros(size=(self.num_channels, max_label+1), device=seg.device, dtype=torch.float32)
        stds_lut = means_lut.clone()

        nlabels = len(class_vals)
        means = sample_uniform((self.num_channels, nlabels), *self.means_bounds, device=seg.device)
        stds = sample_uniform((self.num_channels, nlabels), *self.stds_bounds, device=seg.device)

        means_lut[:, class_vals] = means
        stds_lut[:, class_vals] = stds
        return {'means': means_lut, 'stds': stds_lut}


    def apply(self, data, **params):

        # coerce to int64 for indexing
        seg = data.get(self.segmentation_key).long()
        if seg.ndim >= 3 and seg.shape[0] == 1:
            seg = seg[0]
        out_shape = [self.num_channels, *seg.shape]

        means_lut, stds_lut = params.get('means'), params.get('stds')
        means_map = means_lut[:, seg]
        stds_map = stds_lut[:, seg]
        data[self.image_key] = torch.randn(out_shape, dtype=torch.float32) * stds_map + means_map
        return data


if __name__ == "__main__":
    label = torch.tensor([[0, 1, 0, 2],
                          [1, 2, 3, 0],
                          [0, 0, 2, 0]])
    # label = label.long() # same as to(int64)
    # means = torch.tensor([23.0, -10., 69.0, 67.0, 47.0])
    # print(means[label])
    gmm = SampleGMM(labels=None, means=None, stds=None, num_channels=2, means_bounds=(0, 250), stds_bounds=(0, 50))
    print(gmm(**{'seg': label}))