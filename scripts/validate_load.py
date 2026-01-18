import argparse
from typing import List

from titanic.dataset import load_titanic_data, Passenger


def print_passenger(p: Passenger) -> None:
    def show(name: str, value) -> str:
        t = type(value).__name__
        return f"{name}={value!r} ({t})"

    fields = [
        show("id", p.id),
        show("survived", p.survived),
        show("pclass", p.pclass),
        show("name", p.name),
        show("sex", p.sex),
        show("age", p.age),
        show("sibsp", p.sibsp),
        show("parch", p.parch),
        show("ticket", p.ticket),
        show("fare", p.fare),
        show("cabin", p.cabin),
        show("embarked", p.embarked),
    ]

    print(" | ".join(fields))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate load_titanic_data")
    parser.add_argument("--file", required=True, help="Path to titanic CSV file")
    parser.add_argument("--rows", type=int, default=2, help="Number of rows to print")
    args = parser.parse_args(argv)

    data, stats = load_titanic_data(args.file)
    n = min(args.rows, len(data))
    print(f"Loaded {len(data)} passengers from {args.file}. Printing first {n} rows:")
    for i in range(n):
        print(f"\nRow {i+1}:")
        print_passenger(data[i])

    print("\nStatistics:", stats)


if __name__ == "__main__":
    main()
