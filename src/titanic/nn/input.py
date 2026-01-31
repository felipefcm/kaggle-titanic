from titanic.dataset import ProcessedPassenger

import torch
from torch.utils.data import TensorDataset


def get_torch_dataset(passengers: list[ProcessedPassenger]):
    rows: list[list[float]] = []

    for p in passengers:
        pclass_onehot = _get_pclass_onehot(p)
        sex = 0.0 if p.sex == "male" else 1.0
        rows.append(
            pclass_onehot + [sex, p.age_z, p.age_was_missing, p.sibsp_z, p.parch_z]
        )

    inputs = torch.tensor(rows, dtype=torch.float32)

    expected = torch.tensor(
        [1.0 if p.survived else 0.0 for p in passengers], dtype=torch.float32
    ).reshape(-1, 1)

    return TensorDataset(inputs, expected)


def _get_pclass_onehot(p: ProcessedPassenger):
    onehot = [0.0, 0.0, 0.0]
    onehot[p.pclass - 1] = 1.0
    return onehot
