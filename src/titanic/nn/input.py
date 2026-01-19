from titanic.dataset import Passenger, Stats

import torch
from torch.utils.data import TensorDataset


def get_torch_dataset(passengers: list[Passenger], stats: Stats) -> TensorDataset:
    input = torch.tensor(
        [_passenger_input(p, stats) for p in passengers],
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [1 if p.survived else 0 for p in passengers],
        dtype=torch.float32,
    ).reshape(-1, 1)

    return TensorDataset(input, expected)


def _passenger_input(p: Passenger, stats: Stats) -> list[float]:
    pclass_onehot = _get_pclass_onehot(p)
    sex = 0 if p.sex == "male" else 1

    age = (
        _zscore_normalize(p.age, stats.age_mean, stats.age_std)
        if p.age is not None
        else 0.0
    )

    age_missing = 1.0 if p.age_was_missing else 0.0

    parch_normalized = _zscore_normalize(p.parch, stats.parch_mean, stats.parch_std)
    sibsp_normalized = _zscore_normalize(p.sibsp, stats.sibsp_mean, stats.sibsp_std)

    num_children = parch_normalized if p.age and p.age > 18 else 0
    num_parents = parch_normalized if p.age and p.age <= 18 else 0

    values = pclass_onehot + [
        sex,
        age,
        age_missing,
        num_children,
        num_parents,
        sibsp_normalized,
    ]
    return values


def _get_pclass_onehot(p: Passenger) -> list[float]:
    onehot = [0.0, 0.0, 0.0]
    onehot[p.pclass - 1] = 1.0

    return onehot


def _zscore_normalize(value: float, mean: float, std: float):
    return (value - mean) / std
