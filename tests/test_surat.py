from surat import buat_surat


def test_surat_memuat_nama_dan_istirahat():
    out = buat_surat("Budi", 2, tanggal="2026-07-14")
    assert "Budi" in out
    assert "2 hari" in out
    assert "2026-07-14" in out
    assert "SURAT KETERANGAN SAKIT" in out


def test_surat_dengan_keterangan():
    out = buat_surat("Budi", 1, keterangan="ISPA ringan", tanggal="2026-07-14")
    assert "ISPA ringan" in out


def test_surat_tanggal_default_terisi():
    out = buat_surat("Budi", 1)
    # tanggal default hari ini -> minimal ada pola tahun-bulan
    assert "-" in out
