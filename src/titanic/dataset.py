import csv
import pandas as pd
from dataclasses import dataclass
from typing import Optional, cast


@dataclass
class Passenger:
    id: int = 0
    survived: bool = False
    pclass: int = 0
    name: str = ""
    sex: str = ""
    age: Optional[float] = None
    sibsp: int = 0
    parch: int = 0
    ticket: str = ""
    fare: float = 0.0
    cabin: Optional[str] = None
    embarked: Optional[str] = None


class Stats:
    total_passengers: int


def load_titanic_data(file_path: str) -> list[Passenger]:
    data: list[Passenger] = []
    dt = pd.read_csv(file_path)

    for row in dt.itertuples():
        p = Passenger()
        p.id = cast(int, row.PassengerId)
        p.survived = bool(row.Survived)
        p.pclass = cast(int, row.Pclass)
        p.name = cast(str, row.Name)
        p.sex = cast(str, row.Sex)
        p.age = cast(float, row.Age) if pd.notna(row.Age) else None
        p.sibsp = cast(int, row.SibSp)
        p.parch = cast(int, row.Parch)
        p.ticket = cast(str, row.Ticket)
        p.fare = cast(float, row.Fare)
        p.cabin = cast(str, row.Cabin) if pd.notna(row.Cabin) else None
        p.embarked = cast(str, row.Embarked) if pd.notna(row.Embarked) else None

        data.append(p)

    return data
