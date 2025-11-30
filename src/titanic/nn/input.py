from titanic.dataset import Passenger

import torch
from torch.utils.data import TensorDataset


def get_torch_dataset(passengers: list[Passenger]) -> TensorDataset:
    input = torch.tensor(
        [passenger_input(p) for p in passengers],
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [1 if p.survived else 0 for p in passengers],
        dtype=torch.float32,
    ).reshape(-1, 1)

    return TensorDataset(input, expected)


def passenger_input(p: Passenger) -> list[float]:
    pclass_onehot = _get_pclass_onehot(p)
    sex = 0 if p.sex == "male" else 1
    age = p.age / 80.0 if p.age is not None else 0.0
    age_missing = 1.0 if p.age is None else 0.0

    num_children = p.parch / 9.0 if p.age and p.age > 18 else 0
    num_parents = p.parch / 9.0 if p.age and p.age <= 18 else 0

    values = pclass_onehot + [sex, age, age_missing, num_children, num_parents]
    return values


def _get_pclass_onehot(p: Passenger) -> list[float]:
    onehot = [0.0, 0.0, 0.0]
    onehot[p.pclass - 1] = 1.0

    return onehot
