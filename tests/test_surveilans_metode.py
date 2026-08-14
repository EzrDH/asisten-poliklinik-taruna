"""Uji pemilihan metode deteksi (z-score, Poisson, threshold tetap)."""
from datetime import date, timedelta

import pytest

from surveilans import METODE_TERSEDIA, deteksi_anomali, p_poisson_atas

CONFIG = {"baseline_hari": 14, "z_ambang": 2.0, "c_min": 3, "p_ambang": 0.05,
          "threshold_kasus": 4}
HARI_INI = date(2026, 8, 14)


def _kunjungan(tanggal, blok, gejala):
    return {"tanggal": tanggal, "blok": blok, "kompi": "1",
            "angkatan": "2023", "gejala": [gejala]}


def _data_dengan_lonjakan(jumlah_lonjakan=6):
    """Baseline ~1 kasus/hari, lalu lonjakan pada HARI_INI."""
    data = []
    for i in range(1, 15):
        t = (HARI_INI - timedelta(days=i)).isoformat()
        data.append(_kunjungan(t, "A", "demam"))
    for _ in range(jumlah_lonjakan):
        data.append(_kunjungan(HARI_INI.isoformat(), "A", "demam"))
    return data


def _data_datar():
    """1 kasus/hari termasuk hari ini - tidak ada lonjakan."""
    return [_kunjungan((HARI_INI - timedelta(days=i)).isoformat(), "A", "demam")
            for i in range(15)]


@pytest.mark.parametrize("metode", ["zscore", "robust", "poisson", "threshold"])
def test_setiap_metode_mendeteksi_lonjakan(metode):
    alerts = deteksi_anomali(_data_dengan_lonjakan(), CONFIG,
                             tanggal=HARI_INI.isoformat(), dimensi=["blok"],
                             metode=metode)
    assert any(a["grup"] == "A" and a["gejala"] == "demam" for a in alerts)


@pytest.mark.parametrize("metode", ["zscore", "robust", "poisson", "threshold"])
def test_setiap_metode_diam_pada_data_datar(metode):
    alerts = deteksi_anomali(_data_datar(), CONFIG,
                             tanggal=HARI_INI.isoformat(), dimensi=["blok"],
                             metode=metode)
    assert alerts == []


@pytest.mark.parametrize("metode", ["zscore", "robust", "poisson", "threshold"])
def test_alert_mencantumkan_metode_dan_skor(metode):
    alerts = deteksi_anomali(_data_dengan_lonjakan(), CONFIG,
                             tanggal=HARI_INI.isoformat(), dimensi=["blok"],
                             metode=metode)
    a = alerts[0]
    assert a["metode"] == metode
    assert "skor" in a  # skor pembanding untuk pengurutan


def test_metode_default_adalah_zscore():
    alerts = deteksi_anomali(_data_dengan_lonjakan(), CONFIG,
                             tanggal=HARI_INI.isoformat(), dimensi=["blok"])
    assert alerts[0]["metode"] == "zscore"


def test_metode_tidak_dikenal_ditolak():
    with pytest.raises(ValueError):
        deteksi_anomali(_data_datar(), CONFIG, tanggal=HARI_INI.isoformat(),
                        metode="entah")


def test_metode_tersedia_terdaftar():
    assert set(METODE_TERSEDIA) == {"zscore", "robust", "poisson", "threshold"}


def test_robust_menyertakan_median_dan_mad():
    alerts = deteksi_anomali(_data_dengan_lonjakan(), CONFIG,
                             tanggal=HARI_INI.isoformat(), dimensi=["blok"],
                             metode="robust")
    assert "median" in alerts[0] and "mad" in alerts[0]


def test_robust_tahan_lonjakan_lama_di_baseline():
    """Baseline tercemar wabah lama: z-score klasik tumpul, robust tetap peka.

    Baseline 1 kasus/hari, kecuali dua hari yang melonjak 9 kasus. Lonjakan
    lama itu menaikkan mean & sigma sehingga z-score klasik gagal menandai
    wabah baru, sedangkan median/MAD tidak tergeser.
    """
    data = []
    for i in range(1, 15):
        jumlah = 9 if i in (5, 6) else 1  # wabah lama mencemari jendela
        for _ in range(jumlah):
            data.append(_kunjungan((HARI_INI - timedelta(days=i)).isoformat(),
                                   "A", "demam"))
    for _ in range(6):  # wabah baru hari ini
        data.append(_kunjungan(HARI_INI.isoformat(), "A", "demam"))

    args = dict(tanggal=HARI_INI.isoformat(), dimensi=["blok"])
    assert deteksi_anomali(data, CONFIG, metode="zscore", **args) == []
    assert deteksi_anomali(data, CONFIG, metode="robust", **args) != []


def test_poisson_menyertakan_p_value():
    alerts = deteksi_anomali(_data_dengan_lonjakan(), CONFIG,
                             tanggal=HARI_INI.isoformat(), dimensi=["blok"],
                             metode="poisson")
    assert 0.0 <= alerts[0]["p_value"] <= 1.0
    assert alerts[0]["p_value"] < CONFIG["p_ambang"]


def test_threshold_pakai_ambang_kasus():
    """threshold_kasus=4: 3 kasus tidak memicu, 5 kasus memicu."""
    cfg = dict(CONFIG, c_min=1)
    data3 = _data_dengan_lonjakan(3)
    data5 = _data_dengan_lonjakan(5)
    assert deteksi_anomali(data3, cfg, tanggal=HARI_INI.isoformat(),
                           dimensi=["blok"], metode="threshold") == []
    assert deteksi_anomali(data5, cfg, tanggal=HARI_INI.isoformat(),
                           dimensi=["blok"], metode="threshold") != []


def test_p_poisson_atas_batas():
    assert p_poisson_atas(0, 1.0) == pytest.approx(1.0)
    assert p_poisson_atas(6, 1.0) < p_poisson_atas(2, 1.0)
    assert p_poisson_atas(1, 0.0) == 0.0
