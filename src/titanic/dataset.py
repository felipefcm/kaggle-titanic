import csv

type Passenger = dict[str, int | float | str]


def load_titanic_data(file_path: str) -> list[Passenger]:
    data: list[Passenger] = []

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data
