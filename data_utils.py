from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Klasifikasi Tingkat Kemiskinan di Indonesia.csv"
MODEL_FILE = BASE_DIR / "model_kemiskinan.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"
LABEL_ENCODER_FILE = BASE_DIR / "label_encoder.pkl"
IMAGE_FILE = BASE_DIR / "assets" / "indonesia(1).jpg"

PROVINCE_COL = "Provinsi"
AREA_COL = "Kab/Kota"
TARGET_COL = "Klasifikasi Kemiskinan"
POVERTY_COL = "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)"
IPM_COL = "Indeks Pembangunan Manusia"
UNEMPLOYMENT_COL = "Tingkat Pengangguran Terbuka"

FEATURE_COLUMNS = [
    POVERTY_COL,
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
    IPM_COL,
    "Umur Harapan Hidup (Tahun)",
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
    UNEMPLOYMENT_COL,
    "Tingkat Partisipasi Angkatan Kerja",
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
]

NUMERIC_COLUMNS = FEATURE_COLUMNS + [TARGET_COL]


def load_raw_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_FILE}")

    return pd.read_csv(DATA_FILE, sep=";", skipinitialspace=True)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned = cleaned.dropna(how="all")

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce",
            )

    required_columns = [PROVINCE_COL, AREA_COL, *FEATURE_COLUMNS, TARGET_COL]
    existing_required_columns = [col for col in required_columns if col in cleaned.columns]

    cleaned = cleaned.dropna(subset=existing_required_columns)
    cleaned[PROVINCE_COL] = cleaned[PROVINCE_COL].astype(str).str.strip()
    cleaned[AREA_COL] = cleaned[AREA_COL].astype(str).str.strip()

    return cleaned.reset_index(drop=True)


def load_clean_data() -> pd.DataFrame:
    return clean_data(load_raw_data())


def get_training_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_clean_data()
    return data[FEATURE_COLUMNS], data[TARGET_COL].astype(int)


def find_area(data: pd.DataFrame, prompt: str) -> pd.Series | None:
    prompt_lower = prompt.lower().strip()
    if not prompt_lower:
        return None

    # Cari semua nama daerah yang muncul dalam prompt
    matches = data[data[AREA_COL].str.lower().apply(lambda area: area.lower() in prompt_lower)]
    if not matches.empty:
        # Pilih nama daerah terpanjang agar 'Pasaman Barat' tidak tertukar dengan 'Pasaman'
        matches = matches.assign(match_len=matches[AREA_COL].str.len())
        return matches.sort_values("match_len", ascending=False).iloc[0]

    token_matches = data[data[AREA_COL].str.lower() == prompt_lower]
    if not token_matches.empty:
        return token_matches.iloc[0]

    return None
