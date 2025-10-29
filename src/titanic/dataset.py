import csv


def load_titanic_data(file_path: str) -> list[dict[str, int | float | str]]:
    data: list[dict[str, int | float | str]] = []

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data
