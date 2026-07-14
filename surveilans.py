import math
from collections import defaultdict
from datetime import date, timedelta

DIMENSI_DEFAULT = ["blok", "kompi", "angkatan"]


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


def deteksi_anomali(kunjungan, config, tanggal=None, dimensi=None):
    baseline_hari = config.get("baseline_hari", 14)
    z_ambang = config.get("z_ambang", 2.0)
    c_min = config.get("c_min", 3)
    tanggal = tanggal or _tanggal_default(kunjungan)
    dimensi = dimensi or DIMENSI_DEFAULT

    t = date.fromisoformat(tanggal)
    window = [(t - timedelta(days=i)).isoformat() for i in range(1, baseline_hari + 1)]

    gejala_semua = {g for k in kunjungan for g in _gejala_norm(k)}
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
                n = len(counts) or 1
                mu = sum(counts) / n
                var = sum((c - mu) ** 2 for c in counts) / n
                sigma = math.sqrt(var)
                if sigma > 0:
                    z = (c_t - mu) / sigma
                else:
                    z = float("inf") if c_t > mu else 0.0
                if z >= z_ambang:
                    alerts.append({
                        "dimensi": dim,
                        "grup": grup,
                        "gejala": gejala,
                        "c_t": c_t,
                        "mu": round(mu, 2),
                        "sigma": round(sigma, 2),
                        "z": 999.0 if z == float("inf") else round(z, 2),
                        "tanggal": tanggal,
                    })

    alerts.sort(key=lambda a: a["z"], reverse=True)
    return alerts
