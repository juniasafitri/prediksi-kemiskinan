from data_utils import FEATURE_COLUMNS, load_clean_data, load_raw_data


def main() -> None:
    raw_data = load_raw_data()
    clean_data = load_clean_data()

    print("=== DATA MENTAH ===")
    print(raw_data.dtypes)
    print(f"Jumlah baris: {len(raw_data)}")

    print("\n=== DATA SETELAH CLEANING ===")
    print(clean_data.dtypes)
    print(f"Jumlah baris: {len(clean_data)}")

    print("\n=== CONTOH FITUR MODEL ===")
    print(clean_data[FEATURE_COLUMNS].head())

    print("\n=== JUMLAH NILAI KOSONG ===")
    print(clean_data.isna().sum())


if __name__ == "__main__":
    main()
