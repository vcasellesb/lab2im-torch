# Lab2im/Synthseg in Pytorch

This repository implements a version of SynthSeg/Labels to Image as described in Billot et al. Note that such an implementation has already been done and probably better by `torchio` (this is research code and not guaranteed to work).

However, this repository is motivated by the following points:

* To side-step `tensorflow` (which I am personally not a fan of).
* Adopting `batchgeneratorsv2`'s more lightweight approach to transformations. 
* To have more control over the pipeline (understand it better), and also take the opportunity to learn.

As `batchgeneratorsv2`, this code is designed to be run on the CPU during loading batches for training a neural net for medical image segmentation. Essentially, the main "entrypoints" are `labels_to_image.py` and `_gen_params.py`. The former creates the pipeline using the parameters defined in the latter.

To run the pipeline on a segmentation, you should pass it to its `__call__` method as a keyword argument, as follows:

```python
from lab2im.labels_to_image import LabelsToImage
from lab2im._gen_params import GenerationParams

label = nib.load(...)
# Important: pass segmentations as 4D pytorch tensors
data = torch.from_numpy(np.asarray(label.dataobj))[None]
# Also, either reorient your image to RAS, or pass the following parameter to the pipeline.
# It is used to determine which axis to do flips on (since the human body tends to be only)
# symmetric in the R/L axis
axis_codes = nib.aff2axcodes(label.affine)
input_resolution = label.header.get_zooms()
lab2im = LabelsToImage(GenerationParams())
data = lab2im(**{'seg': data, 'input_resolution': input_resolution, 'axis_codes': axis_codes})
```

That is, each transform gets the data (case, subject in torchio's parlance) as a dictionary, and stores the image and modifies the segmentation using the following keys:
```python3
image_key = 'im'
segmentation_key = 'seg'
```

Although this is subject to change.