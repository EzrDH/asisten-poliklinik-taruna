# Model Card - Detektor Wabah Terlatih

Dihasilkan otomatis oleh `klasifikasi.py`.

## Tujuan
Menjawab secara empiris: apakah model machine learning terlatih mengungguli
surveilans statistik klasik untuk mendeteksi klaster wabah di asrama taruna?

## Data
- Sumber: dataset **sintetis berlabel** (`evaluasi.buat_dataset_berlabel`).
- Label: wabah yang benar-benar disuntikkan - **bukan** keluaran aturan/detektor,
  sehingga tidak terjadi penalaran melingkar.
- Latih: seed [0, 1, 2, 3, 4, 5] - Uji: seed [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] (terpisah, tanpa kebocoran).
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
| ML: rf | 0.784 | 0.944 | **0.856** | 0.967 |
| ML: logreg | 0.698 | 0.944 | **0.802** | 0.952 |
| Statistik: poisson | 0.712 | 0.874 | **0.781** | 0.951 |
| Statistik: threshold | 0.633 | 0.938 | **0.755** | 0.938 |
| Statistik: zscore | 0.741 | 0.745 | **0.740** | 0.947 |
| Statistik: robust | 0.600 | 0.941 | **0.731** | 0.929 |

**Terbaik: ML: rf (F1 0.856).**

Hiperparameter terpilih: `logreg`: {'klas__C': 1.0}, `rf`: {'klas__max_depth': 5, 'klas__n_estimators': 100}

## Ketidakpastian hasil

Satu angka rata-rata bisa menyesatkan. Tabel berikut melaporkan **sebaran F1
antar 10 dataset uji** (rata-rata ± simpangan baku sampel) serta
**ROC-AUC**, yang menilai mutu peringkat tanpa dipengaruhi pemilihan ambang.

| Pendekatan | F1 rata-rata ± SB | Rentang F1 | ROC-AUC |
|------------|-------------------|------------|---------|
| ML: rf | 0.856 ± 0.054 | 0.778 - 0.968 | 0.992 |
| ML: logreg | 0.803 ± 0.063 | 0.710 - 0.909 | 0.991 |
| Statistik: poisson | 0.781 ± 0.081 | 0.667 - 0.938 | 0.981 |
| Statistik: threshold | 0.755 ± 0.042 | 0.692 - 0.821 | 0.985 |
| Statistik: zscore | 0.740 ± 0.096 | 0.600 - 0.903 | 0.969 |
| Statistik: robust | 0.731 ± 0.068 | 0.593 - 0.839 | 0.974 |

## Uji signifikansi (berpasangan)

Membandingkan **ML: rf** dengan **Statistik: poisson** pada dataset uji yang sama
persis (uji-t berpasangan atas F1):

- Selisih rata-rata F1: **+0.075**
- Menang di **7 dari 10** dataset
- t = 3.184, p = 0.0111 -> **signifikan** pada alfa 0,05

Uji dilakukan berpasangan karena kedua pendekatan dinilai pada dataset yang
sama, sehingga variasi antar-dataset tidak mencemari perbandingan. Dengan
n = 10, hasil ini tetap perlu dibaca hati-hati: signifikansi statistik
pada data sintetis bukan jaminan keunggulan di lapangan.

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
