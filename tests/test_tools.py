from datetime import date
import storage
import tools

MASTER = [
    {"no_taruna": "2023045", "nama": "Budi Pratama", "blok": "A",
     "kompi": "2", "angkatan": "2023"},
]


def _setup(tmp_path, kunjungan=None):
    storage.DATA_DIR = tmp_path
    storage.save_master(MASTER)
    storage.save_kunjungan(kunjungan or [])


def test_catat_kunjungan_sukses(tmp_path):
    _setup(tmp_path)
    hasil = tools.catat_kunjungan("Budi Pratama", "demam, batuk", suhu=38.5)
    assert "Budi Pratama" in hasil
    assert "Blok A" in hasil
    assert len(storage.load_kunjungan()) == 1


def test_catat_kunjungan_taruna_tidak_ada(tmp_path):
    _setup(tmp_path)
    hasil = tools.catat_kunjungan("Nama Asing", "demam")
    assert "tidak ditemukan" in hasil.lower()
    assert storage.load_kunjungan() == []


def test_cek_anomali_mendeteksi_lonjakan(tmp_path):
    hari_ini = date.today().isoformat()
    kunjungan = [{"id": i, "tanggal": hari_ini, "no_taruna": "x", "nama": "x",
                  "blok": "A", "kompi": "1", "angkatan": "2023",
                  "gejala": ["demam"], "suhu": 38.0, "urgensi": "x",
                  "catatan": ""} for i in range(6)]
    _setup(tmp_path, kunjungan)
    hasil = tools.cek_anomali(dimensi="blok")
    assert "klaster" in hasil.lower() or "A" in hasil


def test_cek_anomali_dimensi_string_gabungan(tmp_path):
    # LLM kadang mengirim "blok, kompi, angkatan" sebagai satu string.
    hari_ini = date.today().isoformat()
    kunjungan = [{"id": i, "tanggal": hari_ini, "no_taruna": "x", "nama": "x",
                  "blok": "A", "kompi": "1", "angkatan": "2023",
                  "gejala": ["demam"], "suhu": 38.0, "urgensi": "x",
                  "catatan": ""} for i in range(6)]
    _setup(tmp_path, kunjungan)
    hasil = tools.cek_anomali(dimensi="blok, kompi, angkatan")
    assert "klaster" in hasil.lower()


def test_riwayat_pasien(tmp_path):
    hari_ini = date.today().isoformat()
    kunjungan = [{"id": 1, "tanggal": hari_ini, "no_taruna": "2023045",
                  "nama": "Budi Pratama", "blok": "A", "kompi": "2",
                  "angkatan": "2023", "gejala": ["batuk"], "suhu": 37.5,
                  "urgensi": "periksa perawat", "catatan": ""}]
    _setup(tmp_path, kunjungan)
    hasil = tools.riwayat_pasien("Budi Pratama")
    assert "batuk" in hasil


def test_buat_surat_sakit(tmp_path):
    _setup(tmp_path)
    hasil = tools.buat_surat_sakit("Budi Pratama", 2)
    assert "SURAT KETERANGAN SAKIT" in hasil
    assert "2 hari" in hasil
