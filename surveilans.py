"""Surveilans sindromik: deteksi lonjakan (klaster) gejala per grup lokasi.

Tiga metode deteksi tersedia dan dapat dibandingkan secara kuantitatif
(lihat `evaluasi.py`):

- ``zscore``    : anomali statistik terhadap baseline rata-rata bergerak.
- ``robust``    : z-score robust (median + MAD) - tahan terhadap lonjakan lama
                  yang mencemari baseline.
- ``poisson``   : uji probabilitas Poisson satu sisi terhadap baseline.
- ``threshold`` : aturan ambang kasus tetap (baseline pembanding paling naif).

Seluruh perhitungan bersifat deterministik - tidak melibatkan LLM sama sekali.
"""
import math
from collections import defaultdict
from datetime import date, timedelta

DIMENSI_DEFAULT = ["blok", "kompi", "angkatan"]
METODE_TERSEDIA = ["zscore", "robust", "poisson", "threshold"]

# Konstanta skala agar MAD sebanding dengan simpangan baku pada data normal.
SKALA_MAD = 0.6745


def _tanggal_default(kunjungan):
    tgl = [k["tanggal"] for k in kunjungan if k.get("tanggal")]
    return max(tgl) if tgl else date.today().isoformat()


def _gejala_norm(k):
    return [str(g).strip().lower() for g in k.get("gejala", []) if str(g).strip()]


def _per_hari(kunjungan, dim, grup, gejala):
    hasil = defaultdict(int)
    for k in kunjungan:
        if k.get(dim) == grup and gejala in _gejala_norm(k):
            hasil[k["tanggal"]] += 1
    return hasil


def p_poisson_atas(c_t, mu):
    """P(X >= c_t | lambda=mu) untuk distribusi Poisson."""
    if mu <= 0:
        return 1.0 if c_t <= 0 else 0.0
    cdf = 0.0
    for k in range(0, c_t):
        cdf += math.exp(-mu) * mu ** k / math.factorial(k)
    return max(0.0, 1.0 - cdf)


def _statistik_baseline(counts):
    """Kembalikan (mu, sigma) dari daftar hitungan baseline."""
    n = len(counts) or 1
    mu = sum(counts) / n
    var = sum((c - mu) ** 2 for c in counts) / n
    return mu, math.sqrt(var)


def _uji_zscore(c_t, mu, sigma, config, counts=None):
    """Kembalikan (memicu_alarm, info) untuk metode z-score."""
    if sigma > 0:
        z = (c_t - mu) / sigma
    else:
        z = float("inf") if c_t > mu else 0.0
    z_bulat = 999.0 if z == float("inf") else round(z, 2)
    return z >= config.get("z_ambang", 2.0), {"z": z_bulat, "skor": z_bulat}


def _median(nilai):
    urut = sorted(nilai)
    n = len(urut)
    if n == 0:
        return 0.0
    tengah = n // 2
    if n % 2:
        return float(urut[tengah])
    return (urut[tengah - 1] + urut[tengah]) / 2


def _uji_robust(c_t, mu, sigma, config, counts=None):
    """z-score robust: median + MAD, tahan lonjakan lama di jendela baseline.

    Rata-rata dan simpangan baku ikut terangkat bila baseline memuat wabah
    sebelumnya, sehingga wabah baru jadi sulit terdeteksi. Median dan MAD
    tidak mudah tergeser oleh sedikit nilai ekstrem.
    """
    counts = counts or []
    med = _median(counts)
    mad = _median([abs(c - med) for c in counts])
    if mad > 0:
        z = SKALA_MAD * (c_t - med) / mad
    else:
        # MAD nol (baseline sangat datar): pakai selisih absolut sebagai proksi.
        z = float("inf") if c_t > med else 0.0
    z_bulat = 999.0 if z == float("inf") else round(z, 2)
    return z >= config.get("z_ambang", 2.0), {
        "z": z_bulat, "skor": z_bulat, "median": med, "mad": mad,
    }


def _uji_poisson(c_t, mu, sigma, config, counts=None):
    """Kembalikan (memicu_alarm, info) untuk uji Poisson satu sisi."""
    p = p_poisson_atas(c_t, mu)
    # Skor = -log10(p) agar makin kecil p, makin tinggi prioritas alert.
    skor = 999.0 if p <= 0 else round(-math.log10(p), 2)
    return p < config.get("p_ambang", 0.05), {"p_value": p, "skor": skor}


def _uji_threshold(c_t, mu, sigma, config, counts=None):
    """Kembalikan (memicu_alarm, info) untuk ambang kasus tetap."""
    ambang = config.get("threshold_kasus", 4)
    return c_t >= ambang, {"ambang": ambang, "skor": float(c_t)}


_UJI = {
    "zscore": _uji_zscore,
    "robust": _uji_robust,
    "poisson": _uji_poisson,
    "threshold": _uji_threshold,
}


def ringkasan_statistik(kunjungan, dim, grup, gejala, tanggal, config):
    """Statistik mentah satu (tanggal, grup, gejala) terhadap baseline-nya.

    Dipakai bersama oleh detektor statistik dan oleh ekstraksi fitur model
    terlatih (`klasifikasi.py`), agar keduanya melihat informasi yang sama
    dan perbandingannya adil.
    """
    baseline_hari = config.get("baseline_hari", 14)
    t = date.fromisoformat(tanggal)
    window = [(t - timedelta(days=i)).isoformat() for i in range(1, baseline_hari + 1)]

    per_hari = _per_hari(kunjungan, dim, grup, gejala)
    counts = [per_hari.get(d, 0) for d in window]
    c_t = per_hari.get(tanggal, 0)
    mu, sigma = _statistik_baseline(counts)
    med = _median(counts)
    mad = _median([abs(c - med) for c in counts])

    # Total kunjungan grup itu (semua gejala) - membedakan wabah spesifik dari
    # hari yang sekadar ramai.
    total_hari_ini = sum(1 for k in kunjungan
                         if k.get(dim) == grup and k["tanggal"] == tanggal)
    total_baseline = [
        sum(1 for k in kunjungan if k.get(dim) == grup and k["tanggal"] == d)
        for d in window
    ]
    mu_total, _ = _statistik_baseline(total_baseline)

    return {
        "c_t": c_t, "mu": mu, "sigma": sigma, "median": med, "mad": mad,
        "total_hari_ini": total_hari_ini, "mu_total": mu_total,
        "counts": counts,
    }


def deteksi_anomali(kunjungan, config, tanggal=None, dimensi=None, metode="zscore"):
    """Deteksi klaster gejala pada satu tanggal.

    Mengembalikan daftar alert (terurut skor menurun). Tiap alert memuat
    dimensi, grup, gejala, jumlah kasus (c_t), baseline (mu/sigma), metode
    yang dipakai, dan skor pembanding.
    """
    if metode not in _UJI:
        raise ValueError(
            f"Metode '{metode}' tidak dikenal. Pilih salah satu: {METODE_TERSEDIA}"
        )

    baseline_hari = config.get("baseline_hari", 14)
    c_min = config.get("c_min", 3)
    tanggal = tanggal or _tanggal_default(kunjungan)
    dimensi = dimensi or DIMENSI_DEFAULT

    t = date.fromisoformat(tanggal)
    window = [(t - timedelta(days=i)).isoformat() for i in range(1, baseline_hari + 1)]

    gejala_semua = {g for k in kunjungan for g in _gejala_norm(k)}
    uji = _UJI[metode]
    alerts = []

    for dim in dimensi:
        grup_semua = {k.get(dim) for k in kunjungan if k.get(dim)}
        for grup in grup_semua:
            for gejala in gejala_semua:
                per_hari = _per_hari(kunjungan, dim, grup, gejala)
                c_t = per_hari.get(tanggal, 0)
                if c_t < c_min:
                    continue
                counts = [per_hari.get(d, 0) for d in window]
                mu, sigma = _statistik_baseline(counts)
                memicu, info = uji(c_t, mu, sigma, config, counts)
                if not memicu:
                    continue
                alert = {
                    "dimensi": dim,
                    "grup": grup,
                    "gejala": gejala,
                    "c_t": c_t,
                    "mu": round(mu, 2),
                    "sigma": round(sigma, 2),
                    "tanggal": tanggal,
                    "metode": metode,
                }
                alert.update(info)
                alerts.append(alert)

    alerts.sort(key=lambda a: a["skor"], reverse=True)
    return alerts
