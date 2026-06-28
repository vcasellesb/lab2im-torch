import abc
import random


class BaseTransform(abc.ABC):

    image_key = 'im'
    segmentation_key = 'seg'

    def __init__(self):
        """"""

    def __call__(self, **data) -> dict:
        return self.apply(data, **self.get_parameters(data))

    @abc.abstractmethod
    def apply(self, data: dict, **params) -> dict:
        pass

    @abc.abstractmethod
    def get_parameters(self, data: dict) -> dict:
        pass


class RandomTransform:
    def __init__(self, transform: BaseTransform, p: float):
        if not (0. <= p <= 1.):
            raise ValueError(f'{p = } provided is not a valid probability... ')
        self.p = p
        self.transform = transform


    def __call__(self, **data):
        if random.random() < self.p:
            data = self.transform.__call__(**data)
        return data
