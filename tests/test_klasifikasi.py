"""Uji model deteksi wabah terlatih (supervised) dan perbandingannya."""
import pytest

from evaluasi import BLOK, buat_dataset_berlabel
from klasifikasi import (
    NAMA_FITUR,
    SEED_LATIH,
    SEED_UJI,
    bangun_matriks,
    buat_model,
    ekstrak_fitur,
    evaluasi_model,
    format_perbandingan,
    kepentingan_fitur,
    latih_model,
)

SEED_CEPAT = [0]  # satu dataset saja agar test tetap ringan


def _tanggal_terakhir(kunjungan):
    return sorted({k["tanggal"] for k in kunjungan})[-1]


# --- Ekstraksi fitur ------------------------------------------------------

def test_fitur_punya_panjang_tetap():
    kunjungan, _ = buat_dataset_berlabel(seed=0)
    fitur = ekstrak_fitur(kunjungan, "A", _tanggal_terakhir(kunjungan))
    assert len(fitur) == len(NAMA_FITUR)
    assert all(isinstance(v, (int, float)) for v in fitur)


def test_fitur_deterministik():
    kunjungan, _ = buat_dataset_berlabel(seed=0)
    tanggal = _tanggal_terakhir(kunjungan)
    assert ekstrak_fitur(kunjungan, "A", tanggal) == \
        ekstrak_fitur(kunjungan, "A", tanggal)


def test_fitur_ct_naik_saat_kasus_bertambah():
    kunjungan, _ = buat_dataset_berlabel(seed=0)
    tanggal = _tanggal_terakhir(kunjungan)
    sebelum = ekstrak_fitur(kunjungan, "A", tanggal)[0]
    tambahan = [{"id": 9000 + i, "tanggal": tanggal, "blok": "A", "kompi": "1",
                 "angkatan": "2023", "gejala": ["demam"]} for i in range(5)]
    sesudah = ekstrak_fitur(kunjungan + tambahan, "A", tanggal)[0]
    assert sesudah == sebelum + 5


def test_proporsi_dalam_rentang_valid():
    kunjungan, _ = buat_dataset_berlabel(seed=1)
    tanggal = _tanggal_terakhir(kunjungan)
    idx = NAMA_FITUR.index("proporsi")
    for blok in BLOK:
        assert 0.0 <= ekstrak_fitur(kunjungan, blok, tanggal)[idx] <= 1.0


# --- Matriks latih --------------------------------------------------------

def test_matriks_konsisten_dan_berlabel_dua_kelas():
    X, y = bangun_matriks(SEED_CEPAT)
    assert len(X) == len(y) > 0
    assert all(len(baris) == len(NAMA_FITUR) for baris in X)
    assert set(y) == {0, 1}  # ada contoh wabah dan bukan wabah


def test_matriks_bertambah_dengan_lebih_banyak_seed():
    X1, _ = bangun_matriks([0])
    X2, _ = bangun_matriks([0, 1])
    assert len(X2) > len(X1)


def test_seed_latih_dan_uji_tidak_beririsan():
    """Syarat mutlak agar hasil uji tidak bocor."""
    assert set(SEED_LATIH).isdisjoint(set(SEED_UJI))


# --- Konstruksi model -----------------------------------------------------

@pytest.mark.parametrize("nama", ["logreg", "rf"])
def test_buat_model_menyediakan_grid(nama):
    pipa, grid = buat_model(nama)
    assert "klas" in pipa.named_steps
    assert len(grid) > 0


def test_model_tidak_dikenal_ditolak():
    with pytest.raises(ValueError):
        buat_model("model_khayalan")


# --- Pelatihan & evaluasi -------------------------------------------------

def test_latih_model_memilih_hiperparameter():
    model = latih_model("logreg", seeds=SEED_CEPAT, cv=2)
    assert model.best_params_
    assert 0.0 <= model.best_score_ <= 1.0


def test_model_terlatih_mengungguli_tebakan_acak():
    """F1 harus jauh di atas 0 - membuktikan model benar-benar belajar."""
    model = latih_model("logreg", seeds=SEED_CEPAT, cv=2)
    assert evaluasi_model(model, seeds=[100])["f1"] > 0.5


def test_evaluasi_model_mengembalikan_metrik_lengkap():
    model = latih_model("logreg", seeds=SEED_CEPAT, cv=2)
    metrik = evaluasi_model(model, seeds=[100])
    for kunci in ("precision", "recall", "f1", "akurasi", "tp", "fp", "fn", "tn"):
        assert kunci in metrik
    assert metrik["tp"] + metrik["fp"] + metrik["fn"] + metrik["tn"] > 0


def test_kepentingan_fitur_mencakup_semua_fitur():
    model = latih_model("logreg", seeds=SEED_CEPAT, cv=2)
    penting = kepentingan_fitur(model)
    assert len(penting) == len(NAMA_FITUR)
    assert {n for n, _ in penting} == set(NAMA_FITUR)
    besaran = [abs(v) for _, v in penting]
    assert besaran == sorted(besaran, reverse=True)


def test_ct_termasuk_fitur_paling_berpengaruh():
    """Sanity check: jumlah kasus hari ini wajib jadi sinyal utama."""
    model = latih_model("logreg", seeds=SEED_CEPAT, cv=2)
    assert "c_t" in [n for n, _ in kepentingan_fitur(model)[:5]]


# --- Pelaporan ------------------------------------------------------------

def test_format_perbandingan_memuat_kedua_pendekatan():
    statistik = {"poisson": {"precision": 0.7, "recall": 0.8, "f1": 0.75,
                             "akurasi": 0.9}}
    model = {"rf": {"precision": 0.8, "recall": 0.9, "f1": 0.85, "akurasi": 0.95}}
    tabel = format_perbandingan(statistik, model)
    assert "ML: rf" in tabel and "Statistik: poisson" in tabel
    # F1 tertinggi tampil lebih dulu
    assert tabel.index("ML: rf") < tabel.index("Statistik: poisson")
