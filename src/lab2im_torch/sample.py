import torch


def sample_uniform(size, a: float | torch.Tensor = 0., b: float | torch.Tensor = 1., *, dtype = None, device = None):
    """
    :param a: lower bound
    :param b: upper bound
    """
    return torch.rand(size, dtype=dtype, device=device) * (b - a) + a


if __name__ == "__main__":
    lbds = torch.tensor([0., 4., 7.])
    ubds = torch.tensor([3., 6., 10.])
    for _ in range(1000):
        random_sample = sample_uniform(3, lbds, ubds)
        for i in range(len(random_sample)):
            assert lbds[i] <= random_sample[i] < ubds[i]
