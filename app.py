import base64
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import (
    AREA_COL,
    BASE_DIR,
    FEATURE_COLUMNS,
    IMAGE_FILE,
    IPM_COL,
    LABEL_ENCODER_FILE,
    MODEL_FILE,
    POVERTY_COL,
    PROVINCE_COL,
    SCALER_FILE,
    TARGET_COL,
    UNEMPLOYMENT_COL,
    find_area,
    load_clean_data,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_clean_data()


@st.cache_resource
def load_prediction_assets():
    missing_files = [
        path
        for path in [MODEL_FILE, SCALER_FILE, LABEL_ENCODER_FILE]
        if not path.exists()
    ]
    if missing_files:
        return None, None, None, missing_files

    with MODEL_FILE.open("rb") as model_file:
        model = pickle.load(model_file)

    with SCALER_FILE.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with LABEL_ENCODER_FILE.open("rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)

    return model, scaler, label_encoder, []


def image_to_data_uri(path) -> str:
    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


df = get_data()
hero_image = image_to_data_uri(IMAGE_FILE)
px.defaults.template = "plotly_white"


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');

:root {{
    --page-bg: #dae8d9;
    --panel-bg: #eff9f0;
    --surface: rgba(255, 255, 255, 0.85);
    --surface-soft: rgba(236, 249, 236, 0.95);
    --text-main: #0f2c1f;
    --text-muted: #4f6356;
    --border-soft: rgba(62, 121, 64, 0.16);
    --navy: #0f3f2e;
    --accent: #2d9d64;
    --accent-soft: #d7f1db;
    --accent-dark: #19633f;
    --shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
    --radius: 24px;
}}

.stApp {{
    background: linear-gradient(180deg, #c5e5c2 0%, #d3efcf 30%, #e1f5dd 100%);
    color: var(--text-main);
    padding-top: 0;
    overflow-x: hidden;
    font-family: 'Inter', 'Poppins', sans-serif;
}}

.block-container {{
    max-width: 1220px;
    margin: 0 auto;
    padding: 0 20px 3.5rem;
    background: transparent;
}}

.reportview-container .main .block-container {{
    padding-left: 20px;
    padding-right: 20px;
    background: transparent;
}}

[data-testid="stHeader"] {{
    background: transparent;
    height: 0;
}}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{
    display: none;
}}

body, div, span, p, label, li, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong,
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stChatMessage, [data-testid="stMarkdownContainer"] {{
    color: var(--text-main);
    font-family: 'Inter', 'Poppins', sans-serif;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Poppins', 'Inter', sans-serif;
}}

h1 {{
    color: var(--text-main);
    font-size: clamp(2.7rem, 3.6vw, 4.8rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-bottom: 0.75rem;
}}

p, li, label, .stMarkdown, .stTextInput label, .stNumberInput label, .stSelectbox label {{
    font-size: 1rem;
    line-height: 1.75;
}}

.nav-shell {{
    position: sticky;
    top: 12px;
    z-index: 999;
    max-width: 1180px;
    margin: 0 auto 24px;
    padding: 14px 18px;
    background: rgba(22, 73, 44, 0.92);
    border-radius: 28px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.10);
}}

.nav-menu {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 12px;
}}

.stRadio [role="radiogroup"] {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    width: 100%;
    background: rgba(15, 57, 40, 0.14) !important;
    border-radius: 999px;
    border: none;
    padding: 10px 14px;
}}

.stRadio [role="radiogroup"] label {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 14px 24px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.08);
    color: #ffffff;
    font-weight: 700;
    transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
}}

.stRadio [role="radiogroup"] label:hover {{
    transform: translateY(-1px);
    background: rgba(44, 102, 67, 0.9);
}}

.stRadio [role="radiogroup"] input:checked + label {{
    background: var(--green-soft);
    border-color: var(--accent);
    color: var(--green);
}}

.stRadio, .stRadio > div, .stRadio div[role="radiogroup"], .stRadio > div > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}

.stRadio [role="radiogroup"] > label {{
    background: rgba(255, 255, 255, 0.12) !important;
    color: #fff !important;
}}

.stRadio [role="radiogroup"] {{
    gap: 16px !important;
    background: rgba(22, 73, 44, 0.24) !important;
    border-radius: 999px !important;
    padding: 8px 12px !important;
}}

.hero {{
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    width: 100vw;
    min-height: 560px;
    padding: 100px 24px 70px;
    background: linear-gradient(180deg, rgba(6, 27, 27, 0.92), rgba(15, 53, 43, 0.90)),
        url("{hero_image}") center/cover no-repeat;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    overflow: hidden;
}}

.hero::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(8, 61, 49, 0.26), rgba(4, 22, 26, 0.7));
    pointer-events: none;
}}

.hero-content {{
    position: relative;
    z-index: 2;
    max-width: 900px;
    margin: 0 auto;
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 12px 20px;
    margin-bottom: 18px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    color: #f6fbf8;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    border: 1px solid rgba(255, 255, 255, 0.18);
}}

.hero-watermark {{
    position: absolute;
    top: 42px;
    left: 50%;
    transform: translateX(-50%);
    color: rgba(255, 255, 255, 0.08);
    font-size: clamp(4.6rem, 14vw, 8.2rem);
    font-weight: 900;
    letter-spacing: 0.4em;
    line-height: 0.88;
    pointer-events: none;
}}

.hero-title {{
    color: #ffffff;
    font-size: clamp(2.8rem, 4.1vw, 4.6rem);
    font-weight: 800;
    line-height: 1.02;
    margin: 0 auto 20px;
    max-width: 760px;
    text-shadow: 0 28px 60px rgba(0, 0, 0, 0.26);
}}

.hero-copy {{
    color: rgba(255, 255, 255, 0.92);
    font-size: 1.05rem;
    line-height: 1.75;
    max-width: 680px;
    margin: 0 auto 38px;
}}

.hero-cta-wrap {{
    margin-top: 28px;
    display: flex;
    justify-content: center;
}}

.feature-row {{
    max-width: 1180px;
    margin: -88px auto 28px;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 24px;
    padding: 0 12px;
    position: relative;
    z-index: 2;
}}

.feature-card {{
    position: relative;
    background: #ffffff;
    border-radius: 22px;
    min-height: 190px;
    padding: 30px 24px 24px;
    text-align: left;
    box-shadow: var(--shadow);
    border: 1px solid rgba(45, 157, 100, 0.12);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}}

.feature-card:hover {{
    transform: translateY(-6px);
    border-color: rgba(45, 157, 100, 0.24);
    box-shadow: 0 28px 55px rgba(15, 23, 42, 0.12);
}}

.feature-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: rgba(45, 157, 100, 0.12);
    color: var(--accent);
    font-size: 1.2rem;
    margin-bottom: 18px;
}}

.feature-number {{
    color: var(--accent-dark);
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 16px;
}}

.feature-title {{
    color: var(--text-main);
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 14px;
}}

.feature-pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 100px;
    padding: 10px 14px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent-dark);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}}

.content-shell {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 20px 24px 44px;
}}

.page-shell {{
    max-width: 1180px;
    margin: 32px auto 0;
    padding: 28px 26px 44px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 30px;
    box-shadow: 0 20px 48px rgba(15, 23, 42, 0.08);
}}

.page-shell .stBlock {{
    background: transparent !important;
}}

.page-shell h1 {{
    margin-top: 0;
    margin-bottom: 26px;
}}

.chat-container {{
    background: #ffffff;
    border-radius: 20px;
    border: 1px solid rgba(217, 226, 239, 0.95);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    padding: 24px;
    margin-bottom: 28px;
}}

[data-testid="stMetric"] {{
    background: var(--panel-bg);
    border-radius: 18px;
    border: 1px solid var(--border-soft);
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
    padding: 20px;
}}

[data-testid="stMetric"] label,
[data-testid="stMetric"] div {{
    color: var(--text-main);
}}

[data-testid="stMetricValue"] {{
    color: var(--accent-dark);
    font-weight: 900;
}}

.stDataFrame, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{
    background: var(--panel-bg);
    border-radius: 20px;
}}

[data-testid="stPlotlyChart"] {{
    border: 1px solid var(--border-soft);
    padding: 10px;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.05);
}}

input, textarea, [data-baseweb="select"] > div {{
    background: #f8fbf6;
    color: var(--text-main);
    border-color: rgba(45, 157, 100, 0.14);
}}

.stButton button,
.stFormSubmitButton button {{
    background: var(--accent);
    border: 1px solid var(--accent);
    color: #ffffff;
    border-radius: 14px;
    font-weight: 800;
    padding: 0.95rem 1.4rem;
    transition: transform 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
}}

.stButton button:hover,
.stFormSubmitButton button:hover {{
    background: #19633f;
    transform: translateY(-1px);
    border-color: #19633f;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
}}

[data-testid="stChatMessage"] {{
    background: #ffffff;
    border: 1px solid var(--border-soft);
    border-radius: 16px;
}}

.footer {{
    max-width: 1120px;
    margin: 42px auto 32px;
    padding: 24px 22px;
    background: rgba(255,255,255,0.94);
    border-radius: 24px;
    color: var(--text-muted);
    font-size: 0.95rem;
    text-align: center;
    border: 1px solid rgba(45, 157, 100, 0.16);
    box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
}}

.footer a {{
    color: var(--accent-dark);
    text-decoration: none;
    font-weight: 700;
}}

.footer a:hover {{
    color: var(--accent);
}}

@media (max-width: 1024px) {{
    .feature-row {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .hero {{
        padding: 86px 20px 58px;
    }}
}}

@media (max-width: 840px) {{
    .nav-menu {{
        margin: 0 auto 24px;
    }}

    .feature-row {{
        grid-template-columns: 1fr;
    }}

    .hero {{
        padding: 80px 18px 50px;
    }}

    .hero-title {{
        font-size: clamp(2.1rem, 5.5vw, 3.4rem);
    }}

    .hero-copy {{
        font-size: 1rem;
    }}
}}

@media (max-width: 560px) {{
    .block-container {{
        padding-left: 12px;
        padding-right: 12px;
    }}

    .hero {{
        padding: 72px 14px 48px;
    }}

    .hero-kicker {{
        padding: 10px 14px;
        font-size: 0.86rem;
    }}

    .hero-title {{
        font-size: 2.25rem;
    }}

    .feature-card {{
        padding: 22px 18px 20px;
    }}

    .page-shell {{
        margin: 16px auto 0;
        padding: 16px 14px 34px;
        border-radius: 20px;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)
if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

st.markdown(
    """
<div class="nav-shell">
    <div class="nav-menu">
""",
    unsafe_allow_html=True,
)

menu_options = ["Beranda", "Statistik", "Chatbot AI", "Dashboard", "Prediksi", "Dataset"]
current_index = menu_options.index(st.session_state.menu) if st.session_state.menu in menu_options else 0
st.session_state.menu = st.radio(
    "Navigasi",
    menu_options,
    index=current_index,
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown(
    """
    </div>
</div>
""",
    unsafe_allow_html=True,
)

menu = st.session_state.menu
show_debug = False


if menu == "Beranda":




    st.markdown(
        f"""
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">Prediksi Kemiskinan Indonesia</div>
        <div class="hero-watermark">DATA</div>
        <div class="hero-title">Analisis & Prediksi Kemiskinan Daerah</div>
        <div class="hero-copy">
            Dibuat oleh Junia Sahfitri. Jelajahi data kemiskinan, IPM, pengangguran, dan indikator sosial ekonomi
            untuk membantu membaca kondisi daerah secara lebih cepat.
        </div>
    </div>
</div>

<div class="feature-row">
    <div class="feature-card">
        <div class="feature-icon">📌</div>
        <div class="feature-number">{df[AREA_COL].nunique():,}</div>
        <div class="feature-title">Jumlah Daerah</div>
        <div class="feature-pill">DATASET</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🏛️</div>
        <div class="feature-number">{df[PROVINCE_COL].nunique():,}</div>
        <div class="feature-title">Jumlah Provinsi</div>
        <div class="feature-pill">NASIONAL</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-number">{df[POVERTY_COL].mean():.2f}%</div>
        <div class="feature-title">Rata-rata Kemiskinan</div>
        <div class="feature-pill">INDIKATOR</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🌐</div>
        <div class="feature-number">{df[IPM_COL].mean():.2f}</div>
        <div class="feature-title">Rata-rata IPM</div>
        <div class="feature-pill">ANALISIS</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Tombol Streamlit ditempatkan setelah Hero Section selesai dirender.
    # Agar posisinya tetap di tengah, tombol dibungkus dengan st.columns.
    # (CSS disesuaikan agar tampilannya sama seperti tombol hero.)
    st.markdown(
        """
        <style>
        .stButton > button {
            background: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            color: #ffffff !important;
            border-radius: 999px !important;
            font-weight: 950 !important;
            min-height: 58px !important;
            min-width: 240px !important;
            font-size: 1rem !important;
            padding: 0 28px !important;
            box-shadow: 0 18px 38px rgba(255, 138, 29, 0.32) !important;
        }
        .stButton > button:hover {
            background: var(--accent-dark) !important;
            border-color: var(--accent-dark) !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Tombol tepat di tengah Hero section (di bawah deskripsi)
    # Tetap gunakan layout responsif (tanpa position:absolute)
    left, center, right = st.columns([1, 1, 1])
    with center:
        if st.button(
            "🚀 Mulai Analisis",
            key="mulai_analisis_beranda",
            use_container_width=True,
        ):
            st.session_state.menu = "Dashboard"
            st.rerun()

elif menu == "Statistik":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.title("Statistik Utama")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Daerah", f"{df[AREA_COL].nunique():,}")
    col2.metric("Jumlah Provinsi", f"{df[PROVINCE_COL].nunique():,}")
    col3.metric("Rata-rata Kemiskinan", f"{df[POVERTY_COL].mean():.2f}%")
    col4.metric("Rata-rata IPM", f"{df[IPM_COL].mean():.2f}")

    col5, col6 = st.columns(2)
    col5.metric("Kemiskinan Tertinggi", f"{df[POVERTY_COL].max():.2f}%")
    col6.metric("Rata-rata Pengangguran", f"{df[UNEMPLOYMENT_COL].mean():.2f}%")

    top_poverty = df.nlargest(8, POVERTY_COL).sort_values(POVERTY_COL)
    fig = px.bar(
        top_poverty,
        x=POVERTY_COL,
        y=AREA_COL,
        color=PROVINCE_COL,
        orientation="h",
        title="8 Daerah dengan Kemiskinan Tertinggi",
        labels={POVERTY_COL: "Kemiskinan (%)"},
    )
    fig.update_layout(
        paper_bgcolor="#7c8a2f",
        plot_bgcolor="#131313",
        font_color="#0f172a",
        title_font_color="#0f172a",
        legend_title_font_color="#0f172a",
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Chatbot AI":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.title("Chatbot AI")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Halo. Tanyakan data daerah, misalnya: Berapa kemiskinan di Aceh Barat?",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Contoh: Berapa kemiskinan di Aceh Barat?")

    if prompt:
        user_text = (prompt or "").strip()
        st.session_state.messages.append({"role": "user", "content": user_text})

        # --- NLP sederhana berbasis keyword ---
        t = user_text.lower()
        row = find_area(df, user_text)

        # Jika menyebut daerah -> utamakan detail data lokasi terlebih dahulu
        if row is not None:
            answer = f"""
### {row[AREA_COL]}

Provinsi: **{row[PROVINCE_COL]}**

Kemiskinan: **{row[POVERTY_COL]:.2f}%**

IPM: **{row[IPM_COL]:.2f}**

Pengangguran: **{row[UNEMPLOYMENT_COL]:.2f}%**

Insight AI:
Daerah ini perlu dilihat bersama indikator pendidikan, pekerjaan, dan layanan dasar agar keputusan kebijakan lebih tepat.
"""

        # Sapaan
        elif any(
            g in t
            for g in [
                "halo",
                "hallo",
                "haloo",
                "hai",
                "haii",
                "hello",
                "hey",
                "apa kabar",
                "selamat pagi",
                "selamat siang",
                "selamat sore",
                "selamat malam",
                "assalamualaikum",
                "wassalam",
            ]
        ):
            answer = "Halo! 👋 Selamat datang di aplikasi Analisis & Prediksi Kemiskinan Daerah. Tanyakan saja seputar IPM, kemiskinan, dashboard, prediksi, atau dataset."

        # Definisi IPM
        elif "ipm" in t:
            answer = (
                "IPM (Indeks Pembangunan Manusia) adalah ukuran komposit untuk menilai kualitas hidup suatu wilayah, biasanya mencakup kesehatan, pendidikan, dan standar hidup."
            )

        # Perbedaan wilayah dengan nama mirip
        elif ("perbedaan" in t or "beda" in t or "dibandingkan" in t) and "pasaman barat" in t and "pasaman" in t:
            row1 = find_area(df, "Pasaman Barat")
            row2 = find_area(df, "Pasaman")
            if row1 is not None and row2 is not None:
                answer = f"""Perbedaan Pasaman Barat dan Pasaman dari dataset:

- Pasaman Barat: kemiskinan **{row1[POVERTY_COL]:.2f}%**, IPM **{row1[IPM_COL]:.2f}**, pengangguran **{row1[UNEMPLOYMENT_COL]:.2f}%**.
- Pasaman: kemiskinan **{row2[POVERTY_COL]:.2f}%**, IPM **{row2[IPM_COL]:.2f}**, pengangguran **{row2[UNEMPLOYMENT_COL]:.2f}%**.

Pasaman Barat dan Pasaman adalah dua kabupaten berbeda, sehingga nilai kemiskinan dan indikator sosial-ekonominya bisa berbeda meskipun namanya mirip."""
            else:
                answer = "Maaf, saya tidak menemukan data lengkap untuk Pasaman Barat dan/atau Pasaman di dataset."

        elif ("perbedaan" in t or "beda" in t or "dibandingkan" in t) and "solok selatan" in t and "solok" in t:
            row1 = find_area(df, "Solok Selatan")
            row2 = find_area(df, "Solok")
            if row1 is not None and row2 is not None:
                answer = f"""Perbedaan Solok Selatan dan Solok dari dataset:

- Solok Selatan: kemiskinan **{row1[POVERTY_COL]:.2f}%**, IPM **{row1[IPM_COL]:.2f}**, pengangguran **{row1[UNEMPLOYMENT_COL]:.2f}%**.
- Solok: kemiskinan **{row2[POVERTY_COL]:.2f}%**, IPM **{row2[IPM_COL]:.2f}**, pengangguran **{row2[UNEMPLOYMENT_COL]:.2f}%**.

Kota Solok dan Solok Kabupaten adalah wilayah yang berbeda dengan karakteristik sosial-ekonomi tersendiri."""
            else:
                answer = "Maaf, saya tidak menemukan data lengkap untuk Solok Selatan dan/atau Solok di dataset."

        # Pola pertanyaan umum
        elif ("berapa" in t or "berapakah" in t or "berapa persen" in t or "berapa nilai" in t or "berapa tingkat" in t or "berapa banyak" in t) and "kemiskinan" in t:
            answer = (
                f"Rata-rata kemiskinan nasional pada dataset adalah **{df[POVERTY_COL].mean():.2f}%**. Jika Anda ingin data per daerah, sebutkan nama kabupaten/kota, misalnya: Pasaman Barat atau Pasaman."
            )

        # Kemiskinan tertinggi
        elif ("kemiskinan tertinggi" in t or ("tertinggi" in t and "kemiskinan" in t) or ("paling tinggi" in t and "kemiskinan" in t) or "kemiskinan paling tinggi" in t):
            max_row = df.loc[df[POVERTY_COL].idxmax()]
            answer = (
                f"Provinsi/daerah dengan kemiskinan tertinggi adalah **{max_row[AREA_COL]}** (Provinsi: **{max_row[PROVINCE_COL]}**) dengan nilai **{max_row[POVERTY_COL]:.2f}%**."
            )

        # Kemiskinan terendah
        elif ("kemiskinan terendah" in t or ("terendah" in t and "kemiskinan" in t) or ("paling rendah" in t and "kemiskinan" in t) or "kemiskinan paling rendah" in t):
            min_row = df.loc[df[POVERTY_COL].idxmin()]
            answer = (
                f"Provinsi/daerah dengan kemiskinan terendah adalah **{min_row[AREA_COL]}** (Provinsi: **{min_row[PROVINCE_COL]}**) dengan nilai **{min_row[POVERTY_COL]:.2f}%**."
            )

        # Kemiskinan sedang / rata-rata
        elif ("kemiskinan sedang" in t or ("sedang" in t and "kemiskinan" in t) or "rata-rata kemiskinan" in t or "rata rata kemiskinan" in t or "kemiskinan rata-rata" in t):
            answer = (
                f"Rata-rata kemiskinan nasional pada dataset adalah **{df[POVERTY_COL].mean():.2f}%**. Kategori 'sedang' dapat dianggap sebagai nilai di sekitar rata-rata dataset."
            )

        # Definisi kemiskinan
        elif "kemiskinan" in t:
            answer = (
                "Kemiskinan adalah kondisi ketika seseorang/kelompok tidak mampu memenuhi kebutuhan dasar sesuai standar tertentu. Pada dataset ini, kemiskinan ditampilkan dalam persentase (%)."
            )

        # Tanya tentang dashboard
        elif "dashboard" in t or "grafik" in t or "membaca grafik" in t:
            answer = (
                "Dashboard menampilkan ringkasan indikator seperti jumlah daerah, kemiskinan tertinggi, rata-rata IPM, serta visualisasi hubungan IPM dan kemiskinan. Silakan pilih provinsi melalui filter untuk melihat perubahan datanya."
            )

        # Tanya prediksi
        elif "prediksi" in t or "melakukan prediksi" in t:
            answer = (
                "Untuk melakukan prediksi: buka menu **Prediksi**, isi nilai fitur (dengan default median dataset), lalu klik **Prediksi**. Sistem akan menghasilkan kelas tingkat kemiskinan dan, jika tersedia, estimasi keyakinan."
            )

        # Tanya Random Forest
        elif "random forest" in t or "random" in t:
            answer = (
                "Random Forest adalah algoritma klasifikasi berbasis ensemble (sekumpulan decision tree) yang menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting."
            )

        # Tanya dataset
        elif "dataset" in t or "dari mana" in t or "asal" in t:
            answer = (
                "Dataset berasal dari file **Klasifikasi Tingkat Kemiskinan di Indonesia** yang digunakan untuk membangun fitur (IPM, pengangguran, dll.) serta target kelas tingkat kemiskinan."
            )

        # Pertanyaan statistik dari DataFrame
        elif "rata-rata ipm" in t or ("rata" in t and "ipm" in t):
            answer = f"Rata-rata IPM nasional pada dataset adalah **{df[IPM_COL].mean():.2f}**."
        elif "provinsi dengan" in t and "kemiskinan" in t and "tertinggi" in t:
            max_row = df.loc[df[POVERTY_COL].idxmax()]
            answer = f"Provinsi dengan kemiskinan tertinggi adalah **{max_row[AREA_COL]}** (Provinsi: **{max_row[PROVINCE_COL]}**) dengan nilai **{max_row[POVERTY_COL]:.2f}%**."
        elif "provinsi" in t and "ipm" in t and "tertinggi" in t:
            max_row = df.loc[df[IPM_COL].idxmax()]
            answer = f"Provinsi/daerah dengan IPM tertinggi adalah **{max_row[AREA_COL]}** (Provinsi: **{max_row[PROVINCE_COL]}**) dengan nilai **{max_row[IPM_COL]:.2f}**."
        elif "jumlah provinsi" in t:
            answer = f"Jumlah provinsi pada dataset adalah **{df[PROVINCE_COL].nunique():,}**."
        elif "jumlah data" in t or "jumlah" in t and "data" in t:
            answer = f"Jumlah data (baris) pada dataset adalah **{len(df):,}**."

        # Jika menyebut daerah -> gunakan find_area
        else:
            # Jika prompt tidak jelas, tetap coba cari daerah.
            row = find_area(df, user_text)
            if row is None:
                answer = (
                    "Maaf, saya belum memahami pertanyaan tersebut. Silakan tanyakan mengenai kemiskinan, IPM, dashboard, prediksi, dataset, atau penggunaan aplikasi."
                )
            else:
                answer = f"""
### {row[AREA_COL]}

Provinsi: **{row[PROVINCE_COL]}**

Kemiskinan: **{row[POVERTY_COL]:.2f}%**

IPM: **{row[IPM_COL]:.2f}**

Pengangguran: **{row[UNEMPLOYMENT_COL]:.2f}%**

Insight AI:
Daerah ini perlu dilihat bersama indikator pendidikan, pekerjaan, dan layanan dasar agar keputusan kebijakan lebih tepat.
"""

        # Spinner proses berpikir
        with st.spinner("AI sedang menganalisis..."):
            st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


elif menu == "Dashboard":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.title("Dashboard Kemiskinan")
    province_options = ["Semua Provinsi", *sorted(df[PROVINCE_COL].unique())]
    st.markdown("""
<style>

/* BOX SELECTBOX */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border-radius: 8px;
}

/* DROPDOWN LIST (INI YANG HITAM) */
ul[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border-radius: 10px;
}

/* ITEM DI DALAM DROPDOWN */
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

/* HOVER ITEM */
li[role="option"]:hover {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
}

/* LABEL SELECTBOX */
div[data-testid="stSelectbox"] label {
    color: #0F172A;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)
    selected_province = st.selectbox("Filter Provinsi", province_options)

    filtered_df = df if selected_province == "Semua Provinsi" else df[df[PROVINCE_COL] == selected_province]

    col1, col2, col3 = st.columns(3)
    col1.metric("Daerah", f"{len(filtered_df):,}")
    col2.metric("Kemiskinan Tertinggi", f"{filtered_df[POVERTY_COL].max():.2f}%")
    col3.metric("Rata-rata IPM", f"{filtered_df[IPM_COL].mean():.2f}")

    top_poverty = filtered_df.nlargest(10, POVERTY_COL).sort_values(POVERTY_COL)
    fig = px.bar(
        top_poverty,
        x=POVERTY_COL,
        y=AREA_COL,
        color=PROVINCE_COL,
        orientation="h",
        title="Top 10 Kemiskinan Tertinggi",
        labels={POVERTY_COL: "Kemiskinan (%)"},
    )
    fig.update_layout(
        paper_bgcolor="#131311",   # putih keabu lembut (tidak terlalu silau)
        plot_bgcolor="#CDFD0C",    # area grafik tetap putih bersih
        font_color="#D43B15",      # lebih kontras & elegan
        title_font_color="#1052ED",
        legend_title_font_color="#4BEA17",
    )
    st.plotly_chart(fig, width="stretch")

    fig2 = px.scatter(
        filtered_df,
        x=IPM_COL,
        y=POVERTY_COL,
        color=PROVINCE_COL,
        hover_data=[AREA_COL],
        title="Hubungan IPM dan Kemiskinan",
        labels={POVERTY_COL: "Kemiskinan (%)", IPM_COL: "IPM"},
    )
    fig2.update_layout(
        paper_bgcolor="#94ac31",
        plot_bgcolor="#d7dad5",
        font_color="#0f172a",
        title_font_color="#0f172a",
        legend_title_font_color="#0f172a",
    )
    st.plotly_chart(fig2, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


elif menu == "Prediksi":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.title("Prediksi Klasifikasi Kemiskinan")

    model, scaler, label_encoder, missing_files = load_prediction_assets()
    if missing_files:
        st.warning("File model belum lengkap. Jalankan `python train_model.py` dari folder aplikasi.")
        st.write("File yang belum ada:")
        for path in missing_files:
            st.write(f"- {path.name}")
        st.stop()

    medians = df[FEATURE_COLUMNS].median(numeric_only=True)
    user_input = {}

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        for index, column in enumerate(FEATURE_COLUMNS):
            container = col1 if index % 2 == 0 else col2
            user_input[column] = container.number_input(
                column,
                value=float(medians[column]),
                step=0.01,
            )

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_df = pd.DataFrame([user_input], columns=FEATURE_COLUMNS)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_input).max() * 100
            st.success(f"Hasil prediksi: Kelas {prediction_label} dengan keyakinan {probability:.2f}%")
        else:
            st.success(f"Hasil prediksi: Kelas {prediction_label}")
    st.markdown("</div>", unsafe_allow_html=True)


elif menu == "Dataset":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.title("Dataset Kemiskinan")
    st.dataframe(df, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


if show_debug:
    st.write("Lokasi app:", BASE_DIR)
    st.write("File gambar ada:", IMAGE_FILE.exists())
    st.write("Jumlah data:", len(df))
    st.write("Kolom target:", TARGET_COL)
