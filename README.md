# NutriVision Dashboard

Dashboard analisis dampak konsumsi fast food terhadap kesehatan. Proyek capstone Coding Camp 2026 powered by DBS Foundation.

## Dataset
- **Jumlah responden:** 768
- **Sumber:** Synthetic data (Kaggle)

## 4 Pertanyaan Bisnis

| Q | Pertanyaan | Insight Utama |
|---|------------|---------------|
| 1 | Dampak fast food terhadap kesehatan | Perbedaan skor kesehatan kecil (0.05 poin) |
| 2 | Kalori tinggi vs energi rendah | 23.7% responden berisiko empty calories |
| 3 | Analisis per kelompok usia | Usia <20 tahun fast food tertinggi (7x/minggu) |
| 4 | Korelasi fast food, olahraga & BMI | Korelasi mendekati nol (-0.05 & -0.04) |

## Fitur
- Filter interaktif (gender, usia, frekuensi fast food)
- 4 visualisasi + insight
- KPI cards
- Data preview

## Tech Stack
- Python + Streamlit
- Pandas, Matplotlib, Seaborn

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
