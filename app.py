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
    .stApp, [data-testid="stSidebar"] {{ background-color: {BLACK}; }}
    .main-header {{
        background: linear-gradient(135deg, {ORANGE}, {DARK_ORANGE});
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .main-header h1 {{ color: {WHITE}; margin: 0; font-size: 2rem; }}
    .main-header p {{ color: {WHITE}; margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.9rem; }}
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
    .metric-card h4 {{ color: {LIGHT_ORANGE}; margin: 0; font-size: 0.8rem; }}
    .metric-card .value {{ color: {WHITE}; margin: 0.3rem 0 0 0; font-size: 1.6rem; font-weight: bold; }}
    .insight-box {{
        background-color: {DARK_GREY};
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid {ORANGE};
    }}
    .insight-title {{ color: {ORANGE}; font-weight: bold; margin-bottom: 0.5rem; }}
    .insight-text {{ color: {LIGHT_GREY}; margin-bottom: 0.5rem; }}
    .conclusion-text {{ color: {WHITE}; font-weight: bold; margin-top: 0.5rem; }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{ color: {ORANGE} !important; font-weight: 500; }}
    [data-testid="stSidebar"] h2 {{ color: {ORANGE} !important; }}
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {{ color: {LIGHT_GREY} !important; }}
    .stMarkdown p, .stMarkdown li {{ color: {LIGHT_GREY}; }}
    .stButton > button {{ background-color: {ORANGE}; color: {WHITE}; border: none; border-radius: 6px; }}
    .stButton > button:hover {{ background-color: {DARK_ORANGE}; color: {WHITE}; }}
    .dataframe {{ background-color: {DARK_GREY}; color: {LIGHT_GREY}; }}
    .dataframe th {{ background-color: {ORANGE}; color: {WHITE}; }}
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_final.csv')
    return df

df = load_data()

# ==================== HEADER ====================
st.markdown(f"""
<div class="main-header">
    <h1> NutriVision Dashboard</h1>
    <p>Fast-food Detection & Nutrition Intelligence System | Business Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("##  Filter Data")
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
    st.caption(f" Total Data: {len(df)} responden")
    st.caption(" Dashboard by CC26-PSU071")

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
st.markdown('<div class="section-header"> Key Performance Indicators</div>', unsafe_allow_html=True)

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
        <h4> Fast Food per Minggu</h4>
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
        <h4> Skor Kesehatan</h4>
        <div class="value">{avg_health:.1f} / 10</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== Q1 ====================
st.markdown('<div class="section-header"> Q1: Dampak Konsumsi Fast Food terhadap Kesehatan</div>', unsafe_allow_html=True)
st.markdown("*Berapakah rata-rata penurunan skor kesehatan dan persentase peningkatan risiko gangguan pencernaan pada kelompok fast food tinggi (>5x/minggu) dibandingkan rendah (<2x/minggu)?*")

ff_groups = df_filtered['Fast_Food_Group'].unique()
low_group = [g for g in ff_groups if 'rendah' in g.lower() or 'low' in g.lower() or '<2' in g]
high_group = [g for g in ff_groups if 'tinggi' in g.lower() or 'high' in g.lower() or '>5' in g]

if low_group and high_group:
    low_key = low_group[0]
    high_key = high_group[0]
    
    q1_data = df_filtered.groupby('Fast_Food_Group').agg(
        Rata_Skor_Kesehatan=('Overall_Health_Score', 'mean'),
        Persen_Gangguan_Pencernaan=('Digestive_Numeric', lambda x: x.mean() * 100)
    ).loc[[low_key, high_key]]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        bars1 = ax1.bar(q1_data.index, q1_data['Rata_Skor_Kesehatan'], color=[DARK_ORANGE, ORANGE], edgecolor=WHITE)
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
        bars2 = ax2.bar(q1_data.index, q1_data['Persen_Gangguan_Pencernaan'], color=[DARK_ORANGE, ORANGE], edgecolor=WHITE)
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
    
    # Insight & Kesimpulan Q1
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title"> Insight Q1:</div>
        <div class="insight-text">
            • Skor kesehatan kelompok fast food tinggi ({q1_data.loc[high_key, 'Rata_Skor_Kesehatan']:.2f}) vs rendah ({q1_data.loc[low_key, 'Rata_Skor_Kesehatan']:.2f})<br>
            • Gangguan pencernaan kelompok fast food tinggi ({q1_data.loc[high_key, 'Persen_Gangguan_Pencernaan']:.2f}%) vs rendah ({q1_data.loc[low_key, 'Persen_Gangguan_Pencernaan']:.2f}%)<br>
            • Perbedaan skor kesehatan sangat kecil
        </div>
        <div class="conclusion-text"> Kesimpulan: Frekuensi konsumsi fast food dalam dataset ini tidak langsung mempengaruhi gangguan pencernaan maupun penurunan skor kesehatan harian. Ada faktor gaya hidup lain yang lebih berpengaruh.</div>
    </div>
    """, unsafe_allow_html=True)
    
# ==================== Q2 ====================
st.markdown('<div class="section-header"> Q2: Kalori Tinggi vs Energi Rendah</div>', unsafe_allow_html=True)
st.markdown("*Berapa proporsi responden dengan asupan kalori >2500 namun energi <5, dan bagaimana hubungannya dengan frekuensi fast food?*")

total_responden_q2 = len(df_filtered)
jumlah_berisiko_q2 = len(df_filtered[df_filtered['High_Cal_Low_Energy'] == 'Ya'])
persentase_berisiko_q2 = (jumlah_berisiko_q2 / total_responden_q2) * 100 if total_responden_q2 > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4> Total Responden</h4>
        <div class="value">{total_responden_q2}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4> Kalori >2500 & Energi <5</h4>
        <div class="value">{jumlah_berisiko_q2}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4> Proporsi Berisiko</h4>
        <div class="value">{persentase_berisiko_q2:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
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

# Insight & Kesimpulan Q2
st.markdown(f"""
<div class="insight-box">
    <div class="insight-title"> Insight Q2:</div>
    <div class="insight-text">
        • Sebesar <b>{persentase_berisiko_q2:.1f}%</b> dari total responden ({jumlah_berisiko_q2} dari {total_responden_q2} responden) memiliki konsumsi kalori harian tinggi (>2500) namun skor energi rendah (<5)<br>
        • Scatter plot mengidentifikasi bahwa sebagian besar kalori berasal dari empty calories (kalori kosong tinggi lemak dan gula)
    </div>
    <div class="conclusion-text"> Kesimpulan: Kalori tinggi dari fast food tidak dapat diubah tubuh menjadi energi jangka panjang, hanya memberi lonjakan energi sesaat lalu tubuh menjadi cepat lemas (sugar crash/energy crash).</div>
</div>
""", unsafe_allow_html=True)

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
    if not pd.isna(val):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', ha='center', color=WHITE, fontsize=9)
for bar, val in zip(bars2, q3_data['Rata_Kunjungan_Dokter']):
    if not pd.isna(val):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', ha='center', color=WHITE, fontsize=9)

st.pyplot(fig4)

# Insight & Kesimpulan Q3
st.markdown(f"""
<div class="insight-box">
    <div class="insight-title"> Insight Q3:</div>
    <div class="insight-text">
        • Kelompok <b>&lt;20 Tahun</b> memiliki rata-rata fast food tertinggi ({q3_data.loc['<20 Tahun', 'Rata_Fast_Food']:.1f}x/minggu) tapi kunjungan dokter terendah ({q3_data.loc['<20 Tahun', 'Rata_Kunjungan_Dokter']:.2f}x/tahun)<br>
        • Kelompok <b>60+ Tahun</b> memiliki rata-rata fast food terendah ({q3_data.loc['60+ Tahun', 'Rata_Fast_Food']:.1f}x/minggu) tapi kunjungan dokter tertinggi ({q3_data.loc['60+ Tahun', 'Rata_Kunjungan_Dokter']:.2f}x/tahun)<br>
        • Konsumsi fast food cenderung menurun seiring bertambahnya usia, sementara kunjungan dokter cenderung meningkat
    </div>
    <div class="conclusion-text"> Kesimpulan: Tingkat konsumsi fast food mingguan tidak mempengaruhi rata-rata kunjungan dokter per tahun. Faktor usia lebih berpengaruh terhadap kunjungan dokter daripada kebiasaan fast food.</div>
</div>
""", unsafe_allow_html=True)

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

# Insight & Kesimpulan Q4
if not korelasi_matrix.empty:
    korelasi_ff_bmi = korelasi_matrix.loc['Fast_Food_Meals_Per_Week', 'BMI']
    korelasi_olahraga_bmi = korelasi_matrix.loc['Physical_Activity_Hours_Per_Week', 'BMI']
else:
    korelasi_ff_bmi = 0
    korelasi_olahraga_bmi = 0

st.markdown(f"""
<div class="insight-box">
    <div class="insight-title"> Insight Q4:</div>
    <div class="insight-text">
        • Korelasi Fast Food vs BMI: <b>{korelasi_ff_bmi:.2f}</b> ({"positif" if korelasi_ff_bmi > 0 else "negatif"})<br>
        • Korelasi Olahraga vs BMI: <b>{korelasi_olahraga_bmi:.2f}</b> ({"positif" if korelasi_olahraga_bmi > 0 else "negatif"})<br>
        • Nilai koefisien mendekati nol (0) menunjukkan hubungan yang sangat lemah
    </div>
    <div class="conclusion-text"> Kesimpulan: Tidak terdapat korelasi linier atau signifikan secara langsung antara ketiga variabel dalam dataset ini. Tingkat BMI seseorang tidak serta merta dipengaruhi oleh konsumsi fast food yang tinggi maupun tingkat olahraga yang rendah saja, melainkan ada faktor lain yang juga mempengaruhi skor BMI seseorang.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== KESIMPULAN AKHIR ====================
st.markdown('<div class="section-header"> Kesimpulan Akhir & Rekomendasi NutriVision</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style='background: linear-gradient(135deg, {DARK_ORANGE}, {ORANGE}); padding: 1.5rem; border-radius: 15px; margin-top: 1rem;'>
    <h3 style='color: {WHITE};'> Ringkasan 4 Pertanyaan Bisnis:</h3>
    <ul style='color: {WHITE};'>
        <li><b>Q1:</b> Fast food tinggi vs rendah: perbedaan skor kesehatan sangat kecil. Gangguan pencernaan malah lebih rendah di kelompok fast food tinggi. <b>Ada faktor lain yang lebih berpengaruh.</b></li>
        <li><b>Q2:</b> {persentase_berisiko_q2:.1f}% responden memiliki kalori tinggi tapi energi rendah → kemungkinan dari empty calories (fast food).</li>
        <li><b>Q3:</b> Usia &lt;20 tahun: fast food tertinggi ({q3_data.loc['<20 Tahun', 'Rata_Fast_Food']:.1f}x/minggu), kunjungan dokter terendah. Usia 60+: fast food terendah ({q3_data.loc['60+ Tahun', 'Rata_Fast_Food']:.1f}x/minggu), kunjungan dokter tertinggi ({q3_data.loc['60+ Tahun', 'Rata_Kunjungan_Dokter']:.2f}x/tahun). <b>Usia lebih berpengaruh ke kunjungan dokter daripada fast food.</b></li>
        <li><b>Q4:</b> Korelasi fast food vs BMI: {korelasi_ff_bmi:.2f}, olahraga vs BMI: {korelasi_olahraga_bmi:.2f} → <b>tidak ada korelasi signifikan.</b></li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='background-color: {DARK_GREY}; padding: 1.5rem; border-radius: 15px; margin-top: 1rem;'>
    <h3 style='color: {ORANGE};'> Rekomendasi untuk NutriVision:</h3>
    <ul style='color: {LIGHT_GREY};'>
        <li><b>1. Edukasi Empty Calories:</b> Berikan edukasi bahwa kalori dari fast food tidak memberi energi tahan lama (sugar crash)</li>
        <li><b>2. Fokus ke Faktor Lain:</b> Karena fast food tidak berkorelasi langsung dengan BMI, NutriVision perlu mempertimbangkan faktor lain seperti genetika, metabolisme, dan pola tidur</li>
        <li><b>3. Target Usia Muda:</b> Kelompok &lt;20 tahun punya konsumsi fast food tertinggi, perlu jadi target edukasi utama</li>
        <li><b>4. Personalisasi:</b> Setiap individu punya faktor risiko berbeda, perlu pendekatan personal</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== DATA PREVIEW ====================
st.markdown('<div class="section-header"> Data Preview</div>', unsafe_allow_html=True)
st.dataframe(df_filtered.head(20), use_container_width=True)

# ==================== FOOTER ====================
st.markdown(f"""
<div style='background-color: {DARK_GREY}; padding: 1rem; border-radius: 10px; margin-top: 1rem; text-align: center;'>
    <p style='color: {LIGHT_GREY}; margin: 0;'>Dashboard dibuat untuk proyek capstone CC26-PSU071 | Data: dataset_final.csv (768 responden)</p>
</div>
""", unsafe_allow_html=True)
