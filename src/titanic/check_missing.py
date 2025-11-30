import pandas as pd


def main():
    check_missing("./dataset/train.csv")
    check_missing("./dataset/test.csv")


def check_missing(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    print(f"Read {len(df)} rows and {len(df.columns)} columns from {csv_path}\n")

    missing = df.isna().sum()
    pct = (missing / len(df)) * 100

    print("Column | Missing count | Missing %")
    print("-" * 80)

    for col in df.columns:
        miss = int(missing[col])
        percent = pct[col]

        print(f"{col!s:15} | {miss:13d} | {percent:8.2f}%")


if __name__ == "__main__":
    main()
