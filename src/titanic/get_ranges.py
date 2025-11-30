import pandas as pd


def main():
    get_ranges("./dataset/train.csv")
    get_ranges("./dataset/test.csv")


def get_ranges(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    print(f"Read {len(df)} rows and {len(df.columns)} columns from {csv_path}\n")

    print("Column | Min value | Max value")
    print("-" * 50)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        min_val = df[col].min()
        max_val = df[col].max()
        print(f"{col!s:15} | {min_val!s:9} | {max_val!s:9}")


if __name__ == "__main__":
    main()
