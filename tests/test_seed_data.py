from datetime import date
import seed_data
from surveilans import deteksi_anomali

CONFIG = {"baseline_hari": 14, "z_ambang": 2.0, "c_min": 3}


def test_generate_master_jumlah_dan_kunci():
    master = seed_data.generate_master(n=30)
    assert len(master) == 30
    for kunci in ("no_taruna", "nama", "blok", "kompi", "angkatan"):
        assert kunci in master[0]


def test_generate_kunjungan_tidak_kosong():
    master = seed_data.generate_master(n=30)
    kunjungan = seed_data.generate_kunjungan(master, hari=21)
    assert len(kunjungan) > 0
    assert kunjungan[0]["id"] == 1


def test_spike_terdeteksi_oleh_surveilans():
    master = seed_data.generate_master(n=40)
    kunjungan = seed_data.generate_kunjungan(master, hari=21, spike=True)
    hari_ini = max(k["tanggal"] for k in kunjungan)
    alerts = deteksi_anomali(kunjungan, CONFIG, tanggal=hari_ini, dimensi=["blok"])
    assert any(a["grup"] == "A" and a["gejala"] == "demam" for a in alerts)
