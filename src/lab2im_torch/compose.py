from .base import BaseTransform, RandomTransform


class ComposeTransforms:

    def __init__(self,
                 transforms: list[BaseTransform | RandomTransform]):
        self.transforms = transforms

    def __call__(self, **data):
        for transform in self.transforms:
            data = transform.__call__(**data)
        return data
