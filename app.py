import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="NutriVision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== THEME KUSTOM ====================
ORANGE = "#FF6B00"
DARK_ORANGE = "#CC5500"
LIGHT_ORANGE = "#FFA559"
BLACK = "#0D0D0D"
DARK_GREY = "#1E1E1E"
WHITE = "#FFFFFF"
LIGHT_GREY = "#D4D4D4"

# Custom CSS
st.markdown(f"""
<style>
    .stApp, [data-testid="stSidebar"] {{
        background-color: {BLACK};
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {ORANGE}, {DARK_ORANGE});
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .main-header h1 {{
        color: {WHITE};
        margin: 0;
        font-size: 2rem;
    }}
    .main-header p {{
        color: {WHITE};
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }}
    
    .section-header {{
        color: {ORANGE};
        border-bottom: 2px solid {ORANGE};
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: bold;
    }}
    
    .metric-card {{
        background-color: {DARK_GREY};
        padding: 0.8rem;
        border-radius: 10px;
        border-left: 4px solid {ORANGE};
        margin-bottom: 0.5rem;
    }}
    .metric-card h4 {{
        color: {LIGHT_ORANGE};
        margin: 0;
        font-size: 0.8rem;
    }}
    .metric-card .value {{
        color: {WHITE};
        margin: 0.3rem 0 0 0;
        font-size: 1.6rem;
        font-weight: bold;
    }}
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {ORANGE} !important;
        font-weight: 500;
    }}
    
    [data-testid="stSidebar"] h2 {{
        color: {ORANGE} !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {{
        color: {LIGHT_GREY} !important;
    }}
    
    .stMarkdown p, .stMarkdown li {{
        color: {LIGHT_GREY};
    }}
    
    .stButton > button {{
        background-color: {ORANGE};
        color: {WHITE};
        border: none;
        border-radius: 6px;
    }}
    .stButton > button:hover {{
        background-color: {DARK_ORANGE};
        color: {WHITE};
    }}
    
    .dataframe {{
        background-color: {DARK_GREY};
        color: {LIGHT_GREY};
    }}
    .dataframe th {{
        background-color: {ORANGE};
        color: {WHITE};
    }}
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_final.csv')
    return df

df = load_data()

# ==================== DATA CLEANING ====================
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature Engineering
def age_group(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 20:
        return '<20 Tahun'
    elif age < 30:
        return '20-29 Tahun'
    elif age < 45:
        return '30-44 Tahun'
    elif age < 60:
        return '45-59 Tahun'
    else:
        return '60+ Tahun'

df['Age_Group'] = df['Age'].apply(age_group)

df['Fast_Food_Group'] = 'Normal'
df.loc[df['Fast_Food_Meals_Per_Week'] > 5, 'Fast_Food_Group'] = 'Tinggi (>5x)'
df.loc[df['Fast_Food_Meals_Per_Week'] < 2, 'Fast_Food_Group'] = 'Rendah (<2x)'

df['High_Cal_Low_Energy'] = (df['Average_Daily_Calories'] > 2500) & (df['Energy_Level_Score'] < 5)
df['Digestive_Numeric'] = df['Digestive_Issues'].map({'Yes': 1, 'No': 0})

# ==================== HEADER ====================
st.markdown(f"""
<div class="main-header">
    <h1>NutriVision Dashboard</h1>
    <p>Fast-food Detection & Nutrition Intelligence System | Business Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## Filter Data")
    st.markdown("---")
    
    gender_options = ['All'] + sorted(df['Gender'].dropna().unique().tolist())
    gender_filter = st.selectbox("Jenis Kelamin", gender_options)
    
    age_options = ['All'] + ['<20 Tahun', '20-29 Tahun', '30-44 Tahun', '45-59 Tahun', '60+ Tahun']
    age_filter = st.selectbox("Kelompok Usia", age_options)
    
    min_meals = float(df['Fast_Food_Meals_Per_Week'].min())
    max_meals = float(df['Fast_Food_Meals_Per_Week'].max())
    meals_range = st.slider(
        "Frekuensi Fast Food per Minggu",
        min_value=min_meals,
        max_value=max_meals,
        value=(min_meals, max_meals)
    )
    
    st.markdown("---")
    st.caption(f"Total Data: {len(df)} responden")
    st.caption("Dashboard by CC26-PSU071")

# ==================== FILTER DATA ====================
df_filtered = df.copy()

if gender_filter != 'All':
    df_filtered = df_filtered[df_filtered['Gender'] == gender_filter]
if age_filter != 'All':
    df_filtered = df_filtered[df_filtered['Age_Group'] == age_filter]
df_filtered = df_filtered[
    (df_filtered['Fast_Food_Meals_Per_Week'] >= meals_range[0]) &
    (df_filtered['Fast_Food_Meals_Per_Week'] <= meals_range[1])
]

# ==================== KPI CARDS ====================
st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_age = df_filtered['Age'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h4> Rata-rata Usia</h4>
        <div class="value">{avg_age:.1f} tahun</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_meals = df_filtered['Fast_Food_Meals_Per_Week'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h4>Fast Food per Minggu</h4>
        <div class="value">{avg_meals:.1f} kali</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_bmi = df_filtered['BMI'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h4> Rata-rata BMI</h4>
        <div class="value">{avg_bmi:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_health = df_filtered['Overall_Health_Score'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h4>Skor Kesehatan</h4>
        <div class="value">{avg_health:.1f} / 10</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== Q1 ====================
st.markdown('<div class="section-header">Q1: Dampak Konsumsi Fast Food terhadap Kesehatan</div>', unsafe_allow_html=True)
st.markdown("*Berapakah rata-rata penurunan skor kesehatan dan persentase peningkatan risiko gangguan pencernaan pada kelompok fast food tinggi (>5x/minggu) dibandingkan rendah (<2x/minggu)?*")

q1_data = df_filtered.groupby('Fast_Food_Group').agg(
    Rata_Skor_Kesehatan=('Overall_Health_Score', 'mean'),
    Persen_Gangguan_Pencernaan=('Digestive_Numeric', lambda x: x.mean() * 100)
).loc[['Rendah (<2x)', 'Tinggi (>5x)']]

# Warna: ORANGE untuk tinggi, DARK_ORANGE untuk rendah
warna_bars = [DARK_ORANGE, ORANGE]

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    bars1 = ax1.bar(q1_data.index, q1_data['Rata_Skor_Kesehatan'], color=warna_bars, edgecolor=WHITE)
    ax1.set_title('Perbandingan Rata-rata Skor Kesehatan', color=WHITE, fontsize=14)
    ax1.set_xlabel('Kelompok Konsumsi Fast Food', color=LIGHT_GREY)
    ax1.set_ylabel('Skor Kesehatan (0-10)', color=LIGHT_GREY)
    ax1.set_ylim(0, 10)
    ax1.tick_params(colors=LIGHT_GREY)
    ax1.set_facecolor(BLACK)
    fig1.patch.set_facecolor(BLACK)
    for bar, val in zip(bars1, q1_data['Rata_Skor_Kesehatan']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', color=WHITE, fontweight='bold', fontsize=12)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bars2 = ax2.bar(q1_data.index, q1_data['Persen_Gangguan_Pencernaan'], color=warna_bars, edgecolor=WHITE)
    ax2.set_title('Persentase Risiko Gangguan Pencernaan', color=WHITE, fontsize=14)
    ax2.set_xlabel('Kelompok Konsumsi Fast Food', color=LIGHT_GREY)
    ax2.set_ylabel('Persentase Responden (%)', color=LIGHT_GREY)
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors=LIGHT_GREY)
    ax2.set_facecolor(BLACK)
    fig2.patch.set_facecolor(BLACK)
    for bar, val in zip(bars2, q1_data['Persen_Gangguan_Pencernaan']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.2f}%', ha='center', color=WHITE, fontweight='bold', fontsize=12)
    st.pyplot(fig2)

st.markdown("---")

# ==================== Q2 ====================
st.markdown('<div class="section-header"> Q2: Kalori Tinggi vs Energi Rendah</div>', unsafe_allow_html=True)
st.markdown("*Berapa proporsi responden dengan asupan kalori >2500 namun energi <5, dan bagaimana hubungannya dengan frekuensi fast food?*")

total_responden = len(df_filtered)
jumlah_berisiko = len(df_filtered[df_filtered['High_Cal_Low_Energy'] == True])
persentase_berisiko = (jumlah_berisiko / total_responden) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Responden</h4>
        <div class="value">{total_responden}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4> Berisiko</h4>
        <div class="value">{jumlah_berisiko}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4> Persentase</h4>
        <div class="value">{persentase_berisiko:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
# Warna berdasarkan Fast_Food_Group
colors_map = {'Rendah (<2x)': DARK_ORANGE, 'Tinggi (>5x)': ORANGE, 'Normal': LIGHT_ORANGE}
scatter = ax3.scatter(df_filtered['Average_Daily_Calories'], df_filtered['Energy_Level_Score'],
                      c=df_filtered['Fast_Food_Meals_Per_Week'], cmap='Oranges', alpha=0.7, s=60)
ax3.axvline(x=2500, color='red', linestyle='--', linewidth=2, label='Batas Kalori Tinggi (>2500)')
ax3.axhline(y=5, color=DARK_ORANGE, linestyle='--', linewidth=2, label='Batas Energi Rendah (<5)')
ax3.set_title('Sebaran Responden: Kalori vs Tingkat Energi', color=WHITE, fontsize=14)
ax3.set_xlabel('Asupan Kalori Harian', color=LIGHT_GREY)
ax3.set_ylabel('Skor Tingkat Energi (0-10)', color=LIGHT_GREY)
ax3.tick_params(colors=LIGHT_GREY)
ax3.set_facecolor(BLACK)
fig3.patch.set_facecolor(BLACK)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Frekuensi Fast Food per Minggu', color=LIGHT_GREY)
cbar.ax.yaxis.set_tick_params(color=LIGHT_GREY)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=LIGHT_GREY)
st.pyplot(fig3)

st.markdown("---")

# ==================== Q3 ====================
st.markdown('<div class="section-header"> Q3: Analisis per Kelompok Usia</div>', unsafe_allow_html=True)
st.markdown("*Kelompok usia mana yang memiliki frekuensi fast food tertinggi sekaligus kunjungan dokter terbanyak?*")

q3_data = df_filtered.groupby('Age_Group').agg(
    Rata_Fast_Food=('Fast_Food_Meals_Per_Week', 'mean'),
    Rata_Kunjungan_Dokter=('Doctor_Visits_Per_Year', 'mean')
).sort_index()

age_order = ['<20 Tahun', '20-29 Tahun', '30-44 Tahun', '45-59 Tahun', '60+ Tahun']
q3_data = q3_data.reindex(age_order)

fig4, ax4 = plt.subplots(figsize=(10, 6))
x = np.arange(len(q3_data.index))
width = 0.35

bars1 = ax4.bar(x - width/2, q3_data['Rata_Fast_Food'], width, label='Fast Food per Minggu', color=ORANGE, edgecolor=WHITE)
bars2 = ax4.bar(x + width/2, q3_data['Rata_Kunjungan_Dokter'], width, label='Kunjungan Dokter per Tahun', color=DARK_ORANGE, edgecolor=WHITE)

ax4.set_title('Pola Konsumsi Fast Food vs Kunjungan Dokter per Kelompok Usia', color=WHITE, fontsize=14)
ax4.set_xlabel('Kelompok Usia', color=LIGHT_GREY)
ax4.set_ylabel('Nilai Rata-rata', color=LIGHT_GREY)
ax4.set_xticks(x)
ax4.set_xticklabels(q3_data.index, color=LIGHT_GREY, rotation=45)
ax4.tick_params(colors=LIGHT_GREY)
ax4.legend()
ax4.set_facecolor(BLACK)
fig4.patch.set_facecolor(BLACK)

for bar, val in zip(bars1, q3_data['Rata_Fast_Food']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', ha='center', color=WHITE, fontsize=9)
for bar, val in zip(bars2, q3_data['Rata_Kunjungan_Dokter']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', ha='center', color=WHITE, fontsize=9)

st.pyplot(fig4)

st.markdown("---")

# ==================== Q4 ====================
st.markdown('<div class="section-header"> Q4: Korelasi Fast Food & Olahraga terhadap BMI</div>', unsafe_allow_html=True)
st.markdown("*Apakah terdapat korelasi positif antara tingginya fast food dan rendahnya olahraga terhadap tingginya nilai BMI?*")

variabel_korelasi = ['Fast_Food_Meals_Per_Week', 'Physical_Activity_Hours_Per_Week', 'BMI']
korelasi_matrix = df_filtered[variabel_korelasi].corr()

fig5, ax5 = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(korelasi_matrix, dtype=bool))
sns.heatmap(korelasi_matrix, mask=mask, annot=True, cmap='Oranges', fmt='.2f', 
            linewidths=0.5, ax=ax5, vmin=-1, vmax=1,
            cbar_kws={'label': 'Koefisien Korelasi'})
ax5.set_title('Matriks Korelasi: Fast Food, Olahraga, dan BMI', color=WHITE, fontsize=14)
ax5.tick_params(colors=LIGHT_GREY)
ax5.set_facecolor(BLACK)
fig5.patch.set_facecolor(BLACK)
st.pyplot(fig5)

st.markdown("---")

# ==================== DATA PREVIEW ====================
st.markdown('<div class="section-header"> Data Preview</div>', unsafe_allow_html=True)
st.dataframe(df_filtered.head(20), use_container_width=True)

# ==================== FOOTER ====================
st.markdown(f"""
<div style='background-color: {DARK_GREY}; padding: 1rem; border-radius: 10px; margin-top: 1rem; text-align: center;'>
    <p style='color: {LIGHT_GREY}; margin: 0;'>Dashboard dibuat untuk proyek capstone CC26-PSU071 | Data: dataset_inject_noise.csv</p>
</div>
""", unsafe_allow_html=True)