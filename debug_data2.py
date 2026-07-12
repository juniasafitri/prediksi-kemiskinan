from data_utils import load_clean_data, load_raw_data


def main() -> None:
    raw_data = load_raw_data()
    clean_data = load_clean_data()

    empty_rows = raw_data[raw_data.iloc[:, 1:].isna().all(axis=1)]

    print(f"Total baris mentah: {len(raw_data)}")
    print(f"Baris kosong: {len(empty_rows)}")

    print("\n=== Baris ke-100 sampai 110 ===")
    print(raw_data.iloc[100:110])

    print("\n=== Contoh baris kosong ===")
    print(empty_rows.head(10))

    print("\n=== Setelah cleaning ===")
    print(f"Total baris bersih: {len(clean_data)}")
    print(f"Total NaN setelah cleaning: {clean_data.isna().sum().sum()}")
    print(clean_data.head())


if __name__ == "__main__":
    main()
