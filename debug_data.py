from data_utils import AREA_COL, IPM_COL, load_clean_data, load_raw_data


def main() -> None:
    raw_data = load_raw_data()
    clean_data = load_clean_data()

    print("=== CEK DATA SEBELUM KONVERSI ===")
    print(raw_data[[AREA_COL, IPM_COL, "Umur Harapan Hidup (Tahun)"]].head(5))

    print("\n=== CEK DATA SETELAH KONVERSI ===")
    print(clean_data[[AREA_COL, IPM_COL, "Umur Harapan Hidup (Tahun)"]].head(5))

    print("\n=== CEK NaN ===")
    print(clean_data.isna().sum())

    print("\n=== TEST CARI DAERAH ===")
    result = clean_data[clean_data[AREA_COL].str.lower() == "simeulue"]
    if result.empty:
        print("Data tidak ditemukan")
        return

    row = result.iloc[0]
    print(f"Ditemukan: {row[AREA_COL]}")
    print(f"{IPM_COL}: {row[IPM_COL]}")
    print(f"Umur Harapan Hidup (Tahun): {row['Umur Harapan Hidup (Tahun)']}")


if __name__ == "__main__":
    main()
