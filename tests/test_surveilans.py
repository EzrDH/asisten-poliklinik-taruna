from datetime import date, timedelta
from surveilans import deteksi_anomali, p_poisson_atas

CONFIG = {"baseline_hari": 14, "z_ambang": 2.0, "c_min": 3}


def _kunjungan(tanggal, blok, gejala):
    return {"tanggal": tanggal, "blok": blok, "kompi": "1",
            "angkatan": "2023", "gejala": [gejala]}


def test_data_datar_tidak_ada_alert():
    hari_ini = date(2026, 7, 14)
    data = []
    # 1 kasus/ hari selama 15 hari termasuk hari ini -> c_t=1 < c_min
    for i in range(15):
        t = (hari_ini - timedelta(days=i)).isoformat()
        data.append(_kunjungan(t, "A", "demam"))
    alerts = deteksi_anomali(data, CONFIG, tanggal=hari_ini.isoformat())
    assert alerts == []


def test_lonjakan_terdeteksi():
    hari_ini = date(2026, 7, 14)
    data = []
    # baseline ~1 kasus/hari
    for i in range(1, 15):
        t = (hari_ini - timedelta(days=i)).isoformat()
        data.append(_kunjungan(t, "A", "demam"))
    # hari ini melonjak jadi 6 kasus demam di blok A
    for _ in range(6):
        data.append(_kunjungan(hari_ini.isoformat(), "A", "demam"))
    alerts = deteksi_anomali(data, CONFIG, tanggal=hari_ini.isoformat(),
                             dimensi=["blok"])
    assert any(a["grup"] == "A" and a["gejala"] == "demam" for a in alerts)
    a = [x for x in alerts if x["grup"] == "A"][0]
    assert a["c_t"] == 6
    assert a["z"] >= 2.0


def test_poisson_atas_monoton():
    # makin jauh di atas rata-rata, makin kecil probabilitasnya
    assert p_poisson_atas(6, 1.0) < p_poisson_atas(2, 1.0)
