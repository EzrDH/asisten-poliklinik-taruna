"""Deteksi wabah dengan model machine learning terlatih (supervised).

Pertanyaan penelitian
---------------------
**Apakah model yang dilatih dari data mengalahkan surveilans statistik klasik
(z-score / Poisson / ambang tetap) dalam mendeteksi klaster wabah?**

Rancangan agar hasilnya sah
---------------------------
1. **Label bukan berasal dari aturan kita.** Ground truth adalah wabah yang
   benar-benar disuntikkan ke dataset sintetis (`evaluasi.buat_dataset_berlabel`).
   Model tidak sekadar meniru detektor statistik - keduanya dinilai terhadap
   kebenaran yang sama dan independen.
2. **Pemisahan train/test berdasarkan seed dataset.** Dataset latih dan uji
   berasal dari seed berbeda, sehingga tidak ada kebocoran (leakage): model
   diuji pada "wabah" yang belum pernah dilihatnya.
3. **Fitur dihitung dari informasi yang sama** yang tersedia bagi detektor
   statistik (`surveilans.ringkasan_statistik`), sehingga perbandingan adil.
4. **Penalaan hiperparameter** memakai validasi silang pada data latih saja.

Jalankan: ``python klasifikasi.py``
"""
import math

from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluasi import (
    BLOK,
    CONFIG_EVALUASI,
    GEJALA_WABAH,
    buat_dataset_berlabel,
    evaluasi_metode,
    hitung_metrik,
)
from surveilans import METODE_TERSEDIA, p_poisson_atas, ringkasan_statistik

# Seed dataset untuk latih dan uji - terpisah tegas agar tidak bocor.
# Data uji diperbanyak menjadi 10 dataset agar rata-rata, simpangan baku, dan
# uji signifikansi punya daya statistik yang memadai.
SEED_LATIH = [0, 1, 2, 3, 4, 5]
SEED_UJI = list(range(100, 110))

NAMA_FITUR = [
    "c_t",              # jumlah kasus gejala target hari ini
    "mu",               # rata-rata baseline
    "sigma",            # simpangan baku baseline
    "median",           # median baseline (robust)
    "mad",              # median absolute deviation
    "z",                # z-score klasik
    "z_robust",         # z-score robust
    "log_p_poisson",    # -log10 p-value Poisson
    "rasio_mu",         # c_t / (mu + 1) - seberapa berlipat dari biasanya
    "proporsi",         # porsi gejala target dari seluruh kunjungan grup
    "rasio_total",      # total kunjungan hari ini / rata-rata total
]


def ekstrak_fitur(kunjungan, grup, tanggal, config=None, dim="blok",
                  gejala=GEJALA_WABAH):
    """Ubah satu (tanggal, grup) menjadi vektor fitur numerik."""
    config = config or CONFIG_EVALUASI
    s = ringkasan_statistik(kunjungan, dim, grup, gejala, tanggal, config)

    c_t, mu, sigma = s["c_t"], s["mu"], s["sigma"]
    med, mad = s["median"], s["mad"]

    z = (c_t - mu) / sigma if sigma > 0 else float(c_t - mu)
    z_robust = 0.6745 * (c_t - med) / mad if mad > 0 else float(c_t - med)
    p = p_poisson_atas(c_t, mu)
    log_p = 20.0 if p <= 0 else min(20.0, -math.log10(p))
    proporsi = c_t / s["total_hari_ini"] if s["total_hari_ini"] else 0.0
    rasio_total = s["total_hari_ini"] / (s["mu_total"] + 1)

    return [c_t, mu, sigma, med, mad, z, z_robust, log_p,
            c_t / (mu + 1), proporsi, rasio_total]


def _hari_evaluasi(kunjungan, baseline_hari):
    return sorted({k["tanggal"] for k in kunjungan})[baseline_hari:]


def bangun_matriks(seeds, config=None):
    """Kumpulkan (X, y) dari beberapa dataset berlabel."""
    config = config or CONFIG_EVALUASI
    X, y = [], []
    for seed in seeds:
        kunjungan, label = buat_dataset_berlabel(seed=seed)
        for tanggal in _hari_evaluasi(kunjungan, config["baseline_hari"]):
            for blok in BLOK:
                X.append(ekstrak_fitur(kunjungan, blok, tanggal, config))
                y.append(1 if (tanggal, blok) in label else 0)
    return X, y


def buat_model(nama="logreg"):
    """Bangun pipeline model beserta ruang pencarian hiperparameternya."""
    if nama == "logreg":
        pipa = Pipeline([
            ("skala", StandardScaler()),
            ("klas", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        grid = {"klas__C": [0.01, 0.1, 1.0, 10.0]}
    elif nama == "rf":
        pipa = Pipeline([
            ("klas", RandomForestClassifier(random_state=42,
                                            class_weight="balanced")),
        ])
        grid = {"klas__n_estimators": [100, 300],
                "klas__max_depth": [3, 5, None]}
    else:
        raise ValueError(f"Model '{nama}' tidak dikenal. Pilih: logreg, rf")
    return pipa, grid


def latih_model(nama="logreg", seeds=None, cv=5):
    """Latih model dengan penalaan hiperparameter via validasi silang.

    Validasi silang hanya menyentuh data latih; data uji tidak pernah dilihat
    selama pemilihan hiperparameter.
    """
    X, y = bangun_matriks(seeds or SEED_LATIH)
    pipa, grid = buat_model(nama)
    pencari = GridSearchCV(pipa, grid, scoring="f1", cv=cv, n_jobs=1)
    pencari.fit(X, y)
    return pencari


def evaluasi_model(model, seeds=None, config=None):
    """Nilai model pada dataset uji (seed yang belum pernah dilatih)."""
    config = config or CONFIG_EVALUASI
    tp = fp = fn = tn = 0
    for seed in seeds or SEED_UJI:
        kunjungan, label = buat_dataset_berlabel(seed=seed)
        for tanggal in _hari_evaluasi(kunjungan, config["baseline_hari"]):
            for blok in BLOK:
                fitur = ekstrak_fitur(kunjungan, blok, tanggal, config)
                prediksi = int(model.predict([fitur])[0])
                benar = (tanggal, blok) in label
                if prediksi and benar:
                    tp += 1
                elif prediksi and not benar:
                    fp += 1
                elif not prediksi and benar:
                    fn += 1
                else:
                    tn += 1
    return hitung_metrik(tp, fp, fn, tn)


def evaluasi_statistik_pada_uji(seeds=None, config=None):
    """Nilai detektor statistik pada dataset uji yang sama persis."""
    seeds = seeds or SEED_UJI
    hasil = {}
    for metode in METODE_TERSEDIA:
        daftar = []
        for seed in seeds:
            kunjungan, label = buat_dataset_berlabel(seed=seed)
            daftar.append(evaluasi_metode(kunjungan, label, metode, config))
        n = len(daftar)
        hasil[metode] = {
            **{k: sum(d[k] for d in daftar) / n
               for k in ("precision", "recall", "f1", "akurasi")},
            **{k: sum(d[k] for d in daftar) for k in ("tp", "fp", "fn", "tn")},
        }
    return hasil


def metrik_per_seed(model, seeds=None, config=None):
    """Metrik model untuk TIAP dataset uji secara terpisah.

    Melaporkan hasil per dataset (bukan hanya rata-rata) memungkinkan
    penghitungan simpangan baku dan uji berpasangan.
    """
    return [evaluasi_model(model, seeds=[s], config=config)
            for s in (seeds or SEED_UJI)]


def metrik_statistik_per_seed(metode, seeds=None, config=None):
    """Metrik detektor statistik untuk tiap dataset uji secara terpisah."""
    hasil = []
    for seed in (seeds or SEED_UJI):
        kunjungan, label = buat_dataset_berlabel(seed=seed)
        hasil.append(evaluasi_metode(kunjungan, label, metode, config))
    return hasil


def ringkas_metrik(daftar, kunci="f1"):
    """Rata-rata, simpangan baku (sampel), dan rentang satu metrik."""
    nilai = [d[kunci] for d in daftar]
    n = len(nilai)
    rata = sum(nilai) / n if n else 0.0
    if n > 1:
        var = sum((v - rata) ** 2 for v in nilai) / (n - 1)  # simpangan sampel
    else:
        var = 0.0
    return {"mean": rata, "std": math.sqrt(var), "n": n,
            "min": min(nilai) if nilai else 0.0,
            "max": max(nilai) if nilai else 0.0}


def uji_berpasangan(daftar_a, daftar_b, kunci="f1"):
    """Uji-t berpasangan: apakah selisih A - B konsisten, bukan kebetulan?

    Berpasangan karena kedua pendekatan dinilai pada **dataset uji yang sama**,
    sehingga variasi antar-dataset tidak mencemari perbandingan.
    """
    a = [d[kunci] for d in daftar_a]
    b = [d[kunci] for d in daftar_b]
    if len(a) != len(b):
        raise ValueError("Jumlah pasangan tidak sama; perbandingan tidak sah.")

    selisih = [x - y for x, y in zip(a, b)]
    n = len(selisih)
    rata = sum(selisih) / n if n else 0.0
    menang = sum(1 for d in selisih if d > 0)

    t_stat, p_value = ttest_rel(a, b) if n > 1 else (0.0, 1.0)
    if p_value != p_value:  # NaN saat seluruh selisih identik
        t_stat, p_value = 0.0, 1.0

    return {"selisih_rata": rata, "t": float(t_stat), "p": float(p_value),
            "n": n, "menang": menang, "signifikan": bool(p_value < 0.05)}


def _skor_dan_label(seeds, config=None):
    """Kumpulkan fitur & label untuk perhitungan ROC-AUC."""
    config = config or CONFIG_EVALUASI
    X, y = [], []
    for seed in seeds or SEED_UJI:
        kunjungan, label = buat_dataset_berlabel(seed=seed)
        for tanggal in _hari_evaluasi(kunjungan, config["baseline_hari"]):
            for blok in BLOK:
                X.append(ekstrak_fitur(kunjungan, blok, tanggal, config))
                y.append(1 if (tanggal, blok) in label else 0)
    return X, y


def roc_auc_model(model, seeds=None, config=None):
    """ROC-AUC model - mutu peringkat, terlepas dari ambang keputusan."""
    X, y = _skor_dan_label(seeds, config)
    skor = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, skor))


def roc_auc_fitur(nama_fitur, seeds=None, config=None):
    """ROC-AUC bila satu statistik dipakai langsung sebagai skor peringkat.

    Memungkinkan perbandingan model vs statistik klasik tanpa dipengaruhi
    pemilihan ambang.
    """
    if nama_fitur not in NAMA_FITUR:
        raise ValueError(f"Fitur '{nama_fitur}' tidak dikenal.")
    idx = NAMA_FITUR.index(nama_fitur)
    X, y = _skor_dan_label(seeds, config)
    return float(roc_auc_score(y, [baris[idx] for baris in X]))


def kepentingan_fitur(model):
    """Kembalikan bobot/kepentingan fitur agar model tetap dapat dijelaskan."""
    inti = model.best_estimator_.named_steps["klas"]
    if hasattr(inti, "coef_"):
        nilai = inti.coef_[0]
    else:
        nilai = inti.feature_importances_
    pasangan = sorted(zip(NAMA_FITUR, nilai), key=lambda x: abs(x[1]), reverse=True)
    return [(nama, float(v)) for nama, v in pasangan]


def format_perbandingan(hasil_statistik, hasil_model):
    """Tabel Markdown: model terlatih vs detektor statistik pada data uji."""
    baris = [
        "| Pendekatan | Precision | Recall | F1 | Akurasi |",
        "|------------|-----------|--------|----|---------|",
    ]
    gabung = [(f"ML: {n}", m) for n, m in hasil_model.items()]
    gabung += [(f"Statistik: {n}", m) for n, m in hasil_statistik.items()]
    for nama, m in sorted(gabung, key=lambda x: x[1]["f1"], reverse=True):
        baris.append(
            f"| {nama} | {m['precision']:.3f} | {m['recall']:.3f} "
            f"| **{m['f1']:.3f}** | {m['akurasi']:.3f} |"
        )
    return "\n".join(baris)


def format_ketidakpastian(ringkasan, auc):
    """Tabel rata-rata +- simpangan baku F1 antar dataset uji, plus ROC-AUC."""
    baris = [
        "| Pendekatan | F1 rata-rata ± SB | Rentang F1 | ROC-AUC |",
        "|------------|-------------------|------------|---------|",
    ]
    for nama, r in sorted(ringkasan.items(), key=lambda x: x[1]["mean"],
                          reverse=True):
        nilai_auc = f"{auc[nama]:.3f}" if nama in auc else "-"
        baris.append(
            f"| {nama} | {r['mean']:.3f} ± {r['std']:.3f} "
            f"| {r['min']:.3f} - {r['max']:.3f} | {nilai_auc} |"
        )
    return "\n".join(baris)


def main(tulis_laporan=True):
    hasil_model = {}
    penjelasan = {}
    per_seed = {}
    auc = {}
    for nama in ("logreg", "rf"):
        model = latih_model(nama)
        hasil_model[nama] = evaluasi_model(model)
        hasil_model[nama]["param_terbaik"] = model.best_params_
        penjelasan[nama] = kepentingan_fitur(model)[:5]
        per_seed[f"ML: {nama}"] = metrik_per_seed(model)
        auc[f"ML: {nama}"] = roc_auc_model(model)

    hasil_statistik = evaluasi_statistik_pada_uji()
    tabel = format_perbandingan(hasil_statistik, hasil_model)

    # Ketidakpastian & mutu peringkat untuk detektor statistik.
    for metode in METODE_TERSEDIA:
        per_seed[f"Statistik: {metode}"] = metrik_statistik_per_seed(metode)
    auc["Statistik: poisson"] = roc_auc_fitur("log_p_poisson")
    auc["Statistik: zscore"] = roc_auc_fitur("z")
    auc["Statistik: robust"] = roc_auc_fitur("z_robust")
    auc["Statistik: threshold"] = roc_auc_fitur("c_t")

    ringkasan = {nama: ringkas_metrik(d) for nama, d in per_seed.items()}
    tabel_ketidakpastian = format_ketidakpastian(ringkasan, auc)

    # Uji berpasangan: model terbaik melawan detektor statistik terbaik.
    nama_ml = max(("ML: logreg", "ML: rf"), key=lambda n: ringkasan[n]["mean"])
    nama_stat = max((f"Statistik: {m}" for m in METODE_TERSEDIA),
                    key=lambda n: ringkasan[n]["mean"])
    uji = uji_berpasangan(per_seed[nama_ml], per_seed[nama_stat])

    juara = max(
        [(f"ML: {n}", m) for n, m in hasil_model.items()]
        + [(f"Statistik: {n}", m) for n, m in hasil_statistik.items()],
        key=lambda x: x[1]["f1"],
    )

    baris_fitur = "\n".join(
        f"- **{nama}**: " + ", ".join(f"`{f}` ({v:+.2f})" for f, v in daftar)
        for nama, daftar in penjelasan.items()
    )

    laporan = f"""# Model Card - Detektor Wabah Terlatih

Dihasilkan otomatis oleh `klasifikasi.py`.

## Tujuan
Menjawab secara empiris: apakah model machine learning terlatih mengungguli
surveilans statistik klasik untuk mendeteksi klaster wabah di asrama taruna?

## Data
- Sumber: dataset **sintetis berlabel** (`evaluasi.buat_dataset_berlabel`).
- Label: wabah yang benar-benar disuntikkan - **bukan** keluaran aturan/detektor,
  sehingga tidak terjadi penalaran melingkar.
- Latih: seed {SEED_LATIH} - Uji: seed {SEED_UJI} (terpisah, tanpa kebocoran).
- Unit data: pasangan (tanggal, blok); gejala target `{GEJALA_WABAH}`.

## Fitur ({len(NAMA_FITUR)})
{", ".join(f"`{f}`" for f in NAMA_FITUR)}

Seluruh fitur berasal dari informasi yang juga tersedia bagi detektor statistik
(`surveilans.ringkasan_statistik`), agar perbandingan adil.

## Model & penalaan
- `logreg`: Regresi Logistik + penskalaan, `class_weight="balanced"`,
  penalaan `C` lewat GridSearchCV (5-fold, skor F1) - **hanya pada data latih**.
- `rf`: Random Forest, penalaan `n_estimators` & `max_depth`.

## Hasil pada data uji (seed yang tak pernah dilatih)

{tabel}

**Terbaik: {juara[0]} (F1 {juara[1]['f1']:.3f}).**

Hiperparameter terpilih: {", ".join(f"`{n}`: {m['param_terbaik']}" for n, m in hasil_model.items())}

## Ketidakpastian hasil

Satu angka rata-rata bisa menyesatkan. Tabel berikut melaporkan **sebaran F1
antar {len(SEED_UJI)} dataset uji** (rata-rata ± simpangan baku sampel) serta
**ROC-AUC**, yang menilai mutu peringkat tanpa dipengaruhi pemilihan ambang.

{tabel_ketidakpastian}

## Uji signifikansi (berpasangan)

Membandingkan **{nama_ml}** dengan **{nama_stat}** pada dataset uji yang sama
persis (uji-t berpasangan atas F1):

- Selisih rata-rata F1: **{uji['selisih_rata']:+.3f}**
- Menang di **{uji['menang']} dari {uji['n']}** dataset
- t = {uji['t']:.3f}, p = {uji['p']:.4f} -> {"**signifikan** pada alfa 0,05" if uji['signifikan'] else "**belum signifikan** pada alfa 0,05"}

Uji dilakukan berpasangan karena kedua pendekatan dinilai pada dataset yang
sama, sehingga variasi antar-dataset tidak mencemari perbandingan. Dengan
n = {uji['n']}, hasil ini tetap perlu dibaca hati-hati: signifikansi statistik
pada data sintetis bukan jaminan keunggulan di lapangan.

## Fitur paling berpengaruh
{baris_fitur}

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
"""

    print(laporan)
    if tulis_laporan:
        with open("MODEL_CARD.md", "w", encoding="utf-8") as f:
            f.write(laporan)
        print("Model card ditulis ke MODEL_CARD.md")
    return {"model": hasil_model, "statistik": hasil_statistik,
            "ringkasan": ringkasan, "auc": auc, "uji": uji}


if __name__ == "__main__":
    main()
