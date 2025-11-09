from titanic.dataset import Passenger
from torch import Tensor
import torch


def passenger_input(p: Passenger) -> Tensor:
    pclass_onehot = _get_pclass_onehot(p)
    sex = 0 if p["Sex"] == "male" else 1
    age = float(p.get("Age", 0.0)) / 100.0
    sibsp = float(p.get("SibSp", 0.0))
    parch = float(p.get("Parch", 0.0))

    values = pclass_onehot + [sex, age, sibsp, parch]
    return torch.tensor(values, dtype=torch.float32)


def _get_pclass_onehot(p: Passenger) -> list[float]:
    pclass = int(p["Pclass"])

    onehot = [0.0, 0.0, 0.0]
    onehot[pclass - 1] = 1.0

    return onehot
