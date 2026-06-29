import typing as ty
import torch


def construct_lut(source_labels: ty.Sequence[int], destination_labels: ty.Sequence[int]):
    max_label = max(source_labels)
    source_labels = torch.as_tensor(source_labels, dtype=torch.long)
    destination_labels = torch.as_tensor(destination_labels, dtype=torch.long)
    lut = torch.zeros(max_label+1, dtype=torch.long)
    lut[source_labels] = destination_labels
    return lut


if __name__ == "__main__":

    source_labels = [1, 2, 5, 0]
    destination_labels = [2, 3, 4, 0]
    lut = construct_lut(source_labels, destination_labels)
    expected = torch.tensor(
        [0, 2, 3, 0, 0, 4], dtype=torch.long
    )

    assert torch.all(expected == lut)