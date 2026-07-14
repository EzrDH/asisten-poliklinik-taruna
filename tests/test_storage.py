import storage


def test_save_load_kunjungan_roundtrip(tmp_path):
    storage.DATA_DIR = tmp_path
    storage.save_kunjungan([{"id": 1, "nama": "Budi"}])
    assert storage.load_kunjungan() == [{"id": 1, "nama": "Budi"}]


def test_load_kunjungan_default_kosong(tmp_path):
    storage.DATA_DIR = tmp_path
    assert storage.load_kunjungan() == []


def test_tambah_kunjungan_menetapkan_id(tmp_path):
    storage.DATA_DIR = tmp_path
    a = storage.tambah_kunjungan({"nama": "A"})
    b = storage.tambah_kunjungan({"nama": "B"})
    assert a["id"] == 1
    assert b["id"] == 2
    assert len(storage.load_kunjungan()) == 2


def test_load_config_default(tmp_path):
    storage.DATA_DIR = tmp_path
    cfg = storage.load_config()
    assert cfg["baseline_hari"] == 14
    assert cfg["z_ambang"] == 2.0
    assert cfg["c_min"] == 3
