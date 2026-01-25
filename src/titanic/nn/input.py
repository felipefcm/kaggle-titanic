from titanic.dataset import Passenger, Stats

import math
import torch
from torch.utils.data import TensorDataset


def get_torch_dataset(passengers: list[Passenger]) -> TensorDataset:
    rows: list[list[float]] = []

    for p in passengers:
        pclass_onehot = _get_pclass_onehot(p)
        sex = 0.0 if p.sex == "male" else 1.0
        age = float(p.age) if p.age is not None else 0.0
        age_missing = 1.0 if p.age_was_missing else 0.0
        parch = float(p.parch)
        sibsp = float(p.sibsp)

        rows.append(pclass_onehot + [sex, age, age_missing, parch, sibsp])

    inputs = torch.tensor(rows, dtype=torch.float32)

    expected = torch.tensor(
        [1.0 if p.survived else 0.0 for p in passengers], dtype=torch.float32
    ).reshape(-1, 1)

    return TensorDataset(inputs, expected)


def prepare_input(dataset: TensorDataset, stats: Stats) -> TensorDataset:
    raw_inputs, expected = dataset.tensors
    # indices in raw_inputs
    pclass = raw_inputs[:, 0:3]
    sex = raw_inputs[:, 3:4]
    age_raw = raw_inputs[:, 4:5]
    age_missing = raw_inputs[:, 5:6]
    parch_raw = raw_inputs[:, 6:7]
    sibsp_raw = raw_inputs[:, 7:8]

    # Normalizations (preserve float32)
    age_z = (age_raw - stats.age_mean) / stats.age_std
    parch_z = (parch_raw - stats.parch_mean) / stats.parch_std
    sibsp_z = (sibsp_raw - stats.sibsp_mean) / stats.sibsp_std

    # If age was missing, set normalized age to 0.0 (same behavior as previous code)
    missing_mask = age_missing == 1.0
    age_z = torch.where(missing_mask, torch.tensor(0.0, dtype=age_z.dtype), age_z)

    # Determine children/parents splitting using age (requires age present)
    age_present_mask = age_missing == 0.0
    is_adult = age_present_mask & (age_raw > 18.0)
    is_child = age_present_mask & (age_raw <= 18.0)

    num_children = torch.where(
        is_adult, parch_z, torch.tensor(0.0, dtype=parch_z.dtype)
    )
    num_parents = torch.where(is_child, parch_z, torch.tensor(0.0, dtype=parch_z.dtype))

    final_inputs = torch.cat(
        [pclass, sex, age_z, age_missing, num_children, num_parents, sibsp_z], dim=1
    )

    return TensorDataset(final_inputs, expected)


def _get_pclass_onehot(p: Passenger) -> list[float]:
    onehot = [0.0, 0.0, 0.0]
    onehot[p.pclass - 1] = 1.0
    return onehot
