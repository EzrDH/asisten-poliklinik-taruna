# Hasil Evaluasi Detektor Wabah

Dihasilkan otomatis oleh `evaluasi.py`. Seluruh angka berasal dari dataset **sintetis berlabel** - kita mengetahui persis hari & blok mana yang benar-benar wabah, sehingga Precision/Recall/F1 dapat dihitung.

- Dataset uji: **5** seed berbeda, masing-masing 60 hari
- Unit evaluasi: pasangan **(tanggal, blok)**; gejala wabah: `demam`
- Tantangan yang sengaja dibuat: gejala wabah juga muncul sebagai keluhan harian, dan ada **hari ramai tanpa wabah** sebagai sumber alarm palsu.

## 1. Konfigurasi default

| Metode | Precision | Recall | F1 | Akurasi | TP | FP | FN |
|--------|-----------|--------|----|---------|----|----|----|
| `threshold` - Ambang kasus tetap (baseline naif) | 0.710 | 0.955 | 0.809 | 0.956 | 66 | 26 | 4 |
| `poisson` - Uji Poisson satu sisi | 0.715 | 0.933 | 0.800 | 0.955 | 64 | 25 | 6 |
| `robust` - z-score robust (median + MAD) | 0.604 | 0.975 | 0.733 | 0.932 | 67 | 44 | 3 |
| `zscore` - z-score (rata-rata + simpangan baku) | 0.684 | 0.690 | 0.667 | 0.933 | 46 | 22 | 24 |

## 2. Setelah penalaan parameter (3 dataset)

| Metode | Parameter | Nilai terbaik | Precision | Recall | F1 |
|--------|-----------|---------------|-----------|--------|----|
| `threshold` | `threshold_kasus` | 4 | 0.782 | 0.958 | 0.858 |
| `poisson` | `p_ambang` | 0.05 | 0.781 | 0.889 | 0.828 |
| `robust` | `z_ambang` | 2.0 | 0.691 | 0.958 | 0.797 |
| `zscore` | `z_ambang` | 1.0 | 0.697 | 0.889 | 0.775 |

**F1 tertinggi setelah penalaan: `threshold` (0.858) pada threshold_kasus=4.**

## Catatan kejujuran

Angka di atas berlaku pada data sintetis dengan asumsi yang dijelaskan di `evaluasi.py`, bukan pada data poliklinik nyata. Tujuannya membandingkan metode secara adil pada kondisi yang sama, bukan mengklaim akurasi lapangan.
