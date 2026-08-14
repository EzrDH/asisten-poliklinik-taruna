# Model Card - Detektor Wabah Terlatih

Dihasilkan otomatis oleh `klasifikasi.py`.

## Tujuan
Menjawab secara empiris: apakah model machine learning terlatih mengungguli
surveilans statistik klasik untuk mendeteksi klaster wabah di asrama taruna?

## Data
- Sumber: dataset **sintetis berlabel** (`evaluasi.buat_dataset_berlabel`).
- Label: wabah yang benar-benar disuntikkan - **bukan** keluaran aturan/detektor,
  sehingga tidak terjadi penalaran melingkar.
- Latih: seed [0, 1, 2, 3, 4, 5] - Uji: seed [100, 101, 102, 103, 104] (terpisah, tanpa kebocoran).
- Unit data: pasangan (tanggal, blok); gejala target `demam`.

## Fitur (11)
`c_t`, `mu`, `sigma`, `median`, `mad`, `z`, `z_robust`, `log_p_poisson`, `rasio_mu`, `proporsi`, `rasio_total`

Seluruh fitur berasal dari informasi yang juga tersedia bagi detektor statistik
(`surveilans.ringkasan_statistik`), agar perbandingan adil.

## Model & penalaan
- `logreg`: Regresi Logistik + penskalaan, `class_weight="balanced"`,
  penalaan `C` lewat GridSearchCV (5-fold, skor F1) - **hanya pada data latih**.
- `rf`: Random Forest, penalaan `n_estimators` & `max_depth`.

## Hasil pada data uji (seed yang tak pernah dilatih)

| Pendekatan | Precision | Recall | F1 | Akurasi |
|------------|-----------|--------|----|---------|
| ML: rf | 0.827 | 0.931 | **0.876** | 0.972 |
| ML: logreg | 0.731 | 0.944 | **0.824** | 0.958 |
| Statistik: poisson | 0.713 | 0.848 | **0.773** | 0.949 |
| Statistik: threshold | 0.637 | 0.942 | **0.760** | 0.939 |
| Statistik: zscore | 0.750 | 0.752 | **0.748** | 0.949 |
| Statistik: robust | 0.576 | 0.922 | **0.709** | 0.923 |

**Terbaik: ML: rf (F1 0.876).**

Hiperparameter terpilih: `logreg`: {'klas__C': 1.0}, `rf`: {'klas__max_depth': 5, 'klas__n_estimators': 100}

## Fitur paling berpengaruh
- **logreg**: `c_t` (+2.60), `rasio_mu` (+1.70), `rasio_total` (-1.31), `proporsi` (+0.92), `log_p_poisson` (-0.91)
- **rf**: `c_t` (+0.23), `rasio_mu` (+0.21), `log_p_poisson` (+0.20), `proporsi` (+0.14), `z_robust` (+0.11)

## Keterbatasan (dibaca sebelum menyimpulkan)
- Hasil berlaku pada **data sintetis** dengan asumsi di `evaluasi.py`, bukan data
  poliklinik nyata. Angka ini **tidak** boleh dibaca sebagai akurasi lapangan.
- Model memerlukan data historis berlabel untuk dilatih; di lapangan label wabah
  jarang tersedia rapi, sehingga detektor statistik tetap dipertahankan sebagai
  metode default sistem (tidak butuh pelatihan, langsung jalan).
- Tidak ada uji pada pergeseran distribusi (musim, perubahan populasi).

## Penggunaan yang dimaksudkan
Alat bantu **peringatan dini** bagi petugas klinik. Bukan alat diagnosis, dan
tidak menggantikan penilaian tenaga kesehatan.
