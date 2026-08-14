"""Uji harness evaluasi kuantitatif detektor wabah."""
import pytest

from evaluasi import (
    CONFIG_EVALUASI,
    PARAM_SWEEP,
    bandingkan_metode,
    buat_dataset_berlabel,
    cari_konfigurasi_terbaik,
    evaluasi_metode,
    format_tabel,
    format_tabel_tuning,
    hitung_metrik,
    sweep_parameter,
)


# --- Metrik dasar ---------------------------------------------------------

def test_hitung_metrik_kasus_umum():
    m = hitung_metrik(tp=8, fp=2, fn=4, tn=86)
    assert m["precision"] == pytest.approx(0.8)       # 8 / 10
    assert m["recall"] == pytest.approx(2 / 3)        # 8 / 12
    assert m["f1"] == pytest.approx(0.7272727, rel=1e-4)
    assert m["akurasi"] == pytest.approx(0.94)        # 94 / 100


def test_hitung_metrik_sempurna():
    m = hitung_metrik(tp=10, fp=0, fn=0, tn=90)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_hitung_metrik_tanpa_prediksi_positif():
    """Tidak ada alert sama sekali: precision didefinisikan 0, bukan error."""
    m = hitung_metrik(tp=0, fp=0, fn=5, tn=95)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_hitung_metrik_menyimpan_hitungan_mentah():
    m = hitung_metrik(tp=1, fp=2, fn=3, tn=4)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 2, 3, 4)


# --- Dataset berlabel -----------------------------------------------------

def test_dataset_punya_kunjungan_dan_label():
    kunjungan, label = buat_dataset_berlabel(seed=1)
    assert len(kunjungan) > 0
    assert len(label) > 0
    # label = himpunan (tanggal, grup) yang benar-benar wabah
    tanggal, grup = next(iter(label))
    assert isinstance(tanggal, str) and len(tanggal) == 10
    assert isinstance(grup, str)


def test_dataset_deterministik_dengan_seed_sama():
    a1, l1 = buat_dataset_berlabel(seed=7)
    a2, l2 = buat_dataset_berlabel(seed=7)
    assert a1 == a2
    assert l1 == l2


def test_dataset_berbeda_untuk_seed_berbeda():
    _, l1 = buat_dataset_berlabel(seed=1)
    _, l2 = buat_dataset_berlabel(seed=2)
    assert l1 != l2


def test_setiap_kunjungan_punya_skema_lengkap():
    kunjungan, _ = buat_dataset_berlabel(seed=3)
    k = kunjungan[0]
    for kunci in ("tanggal", "blok", "kompi", "angkatan", "gejala"):
        assert kunci in k
    assert isinstance(k["gejala"], list)


def test_hari_wabah_punya_kasus_lebih_banyak():
    """Sanity check: hari berlabel wabah memang punya lonjakan kasus."""
    kunjungan, label = buat_dataset_berlabel(seed=5)
    tanggal_wabah, blok_wabah = next(iter(label))
    kasus_wabah = sum(
        1 for k in kunjungan
        if k["tanggal"] == tanggal_wabah and k["blok"] == blok_wabah
    )
    assert kasus_wabah >= CONFIG_EVALUASI["c_min"]


# --- Evaluasi per metode --------------------------------------------------

@pytest.mark.parametrize("metode", ["zscore", "robust", "poisson", "threshold"])
def test_evaluasi_metode_mengembalikan_metrik_valid(metode):
    kunjungan, label = buat_dataset_berlabel(seed=11)
    m = evaluasi_metode(kunjungan, label, metode)
    for kunci in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
        assert kunci in m
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0


def test_evaluasi_mendeteksi_sebagian_besar_wabah():
    """z-score harus menangkap mayoritas wabah yang disuntikkan."""
    kunjungan, label = buat_dataset_berlabel(seed=13)
    m = evaluasi_metode(kunjungan, label, "zscore")
    assert m["recall"] >= 0.5


def test_evaluasi_metode_tidak_dikenal_ditolak():
    kunjungan, label = buat_dataset_berlabel(seed=2)
    with pytest.raises(ValueError):
        evaluasi_metode(kunjungan, label, "metode_khayalan")


# --- Perbandingan lintas metode -------------------------------------------

def test_bandingkan_metode_mencakup_semua_metode():
    hasil = bandingkan_metode(n_dataset=2)
    assert set(hasil) == {"zscore", "robust", "poisson", "threshold"}
    for metrik in hasil.values():
        assert 0.0 <= metrik["f1"] <= 1.0
        assert metrik["n_dataset"] == 2


# --- Penalaan parameter (hyperparameter tuning) ---------------------------

@pytest.mark.parametrize("metode", ["zscore", "robust", "poisson", "threshold"])
def test_sweep_menguji_setiap_kandidat(metode):
    hasil = sweep_parameter(metode, n_dataset=1)
    nama, kandidat = PARAM_SWEEP[metode]
    assert len(hasil) == len(kandidat)
    assert [h["nilai"] for h in hasil] == kandidat
    assert all(h["parameter"] == nama for h in hasil)


def test_sweep_metode_tanpa_parameter_ditolak():
    with pytest.raises(ValueError):
        sweep_parameter("metode_khayalan", n_dataset=1)


def test_ambang_lebih_longgar_menaikkan_recall():
    """Semakin rendah z_ambang, semakin banyak alert -> recall tidak menurun."""
    hasil = sweep_parameter("zscore", n_dataset=1)
    urut = sorted(hasil, key=lambda h: h["nilai"])
    assert urut[0]["recall"] >= urut[-1]["recall"]


def test_konfigurasi_terbaik_punya_f1_tertinggi():
    terbaik = cari_konfigurasi_terbaik(n_dataset=1)
    assert set(terbaik) == set(PARAM_SWEEP)
    for metode, pilihan in terbaik.items():
        semua = sweep_parameter(metode, n_dataset=1)
        assert pilihan["f1"] == max(h["f1"] for h in semua)


def test_format_tabel_tuning_memuat_parameter():
    terbaik = cari_konfigurasi_terbaik(n_dataset=1)
    tabel = format_tabel_tuning(terbaik)
    assert "Parameter" in tabel and "Nilai terbaik" in tabel
    assert "z_ambang" in tabel and "threshold_kasus" in tabel


def test_format_tabel_memuat_header_dan_metode():
    hasil = bandingkan_metode(n_dataset=1)
    tabel = format_tabel(hasil)
    assert "Precision" in tabel and "Recall" in tabel and "F1" in tabel
    for metode in ("zscore", "robust", "poisson", "threshold"):
        assert metode in tabel
