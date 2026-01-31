import pandas as pd
import numpy as np
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
    age_was_missing: bool = False


@dataclass
class ProcessedPassenger(Passenger):
    age_z: float = 0.0
    sibsp_z: float = 0.0
    parch_z: float = 0.0


@dataclass
class Stats:
    total_passengers: int
    age_by_class_sex: dict[int, dict[str, float]]

    age_mean: float = 0.0
    age_std: float = 1.0
    sibsp_mean: float = 0.0
    sibsp_std: float = 1.0
    parch_mean: float = 0.0
    parch_std: float = 1.0

    age_z: float = 0.0
    sibsp_z: float = 0.0
    parch_z: float = 0.0


def load_titanic_data(file_path: str):
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
        p.age_was_missing = p.age is None
        p.sibsp = cast(int, row.SibSp)
        p.parch = cast(int, row.Parch)
        p.ticket = cast(str, row.Ticket)
        p.fare = cast(float, row.Fare)
        p.cabin = cast(str, row.Cabin) if pd.notna(row.Cabin) else None
        p.embarked = cast(str, row.Embarked) if pd.notna(row.Embarked) else None

        data.append(p)

    return data


def process_passengers(passengers: list[Passenger]):
    stats = _compute_stats(passengers)
    passengers = _impute_missing_ages(passengers, stats.age_by_class_sex)
    processed_passengers = _normalise_passengers(passengers, stats)

    return (processed_passengers, stats)


def _compute_stats(passengers: list[Passenger]):
    ages = [p.age for p in passengers if p.age is not None]
    parchs = [p.parch for p in passengers]
    sibsps = [p.sibsp for p in passengers]

    stats = Stats(
        total_passengers=len(passengers),
        age_by_class_sex=_compute_age_means_by_class_sex(passengers),
        age_mean=float(np.mean(ages)) if ages else 0.0,
        age_std=float(np.std(ages)) if ages else 1.0,
        parch_mean=float(np.mean(parchs)),
        parch_std=float(np.std(parchs)) if np.std(parchs) > 0 else 1.0,
        sibsp_mean=float(np.mean(sibsps)),
        sibsp_std=float(np.std(sibsps)) if np.std(sibsps) > 0 else 1.0,
    )

    return stats


def _compute_age_means_by_class_sex(
    passengers: list[Passenger],
) -> dict[int, dict[str, float]]:
    age_by_class_sex: dict[int, dict[str, float]] = {}

    for pclass in [1, 2, 3]:
        age_by_class_sex[pclass] = {}
        for sex in ["male", "female"]:
            ages = [
                p.age
                for p in passengers
                if p.pclass == pclass and p.sex == sex and p.age is not None
            ]
            age_by_class_sex[pclass][sex] = float(np.mean(ages)) if ages else 0.0

    return age_by_class_sex


def _impute_missing_ages(
    passengers: list[Passenger], age_by_class_sex: dict[int, dict[str, float]]
) -> list[Passenger]:
    imputed_passengers = []

    for p in passengers:
        if p.age is None:
            p.age = age_by_class_sex.get(p.pclass, {}).get(p.sex, 0.0)
        imputed_passengers.append(p)

    return imputed_passengers


def _normalise_passengers(passengers: list[Passenger], stats: Stats):
    processed_passengers: list[ProcessedPassenger] = []

    for p in passengers:
        if p.age is None:
            raise ValueError("Passenger age should not be None when normalising")

        processed_passenger: ProcessedPassenger = cast(ProcessedPassenger, p)

        processed_passenger.age_z = (p.age - stats.age_mean) / stats.age_std
        processed_passenger.sibsp_z = (p.sibsp - stats.sibsp_mean) / stats.sibsp_std
        processed_passenger.parch_z = (p.parch - stats.parch_mean) / stats.parch_std

        processed_passengers.append(processed_passenger)

    return processed_passengers
