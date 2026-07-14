from triase import triase, DISCLAIMER


def test_gejala_darurat_segera():
    hasil = triase(["sesak napas"])
    assert hasil["urgensi"] == "segera"
    assert hasil["disclaimer"] == DISCLAIMER


def test_suhu_sangat_tinggi_segera():
    assert triase(["batuk"], suhu=39.5)["urgensi"] == "segera"


def test_demam_periksa_dokter():
    assert triase(["batuk", "pilek"], suhu=38.2)["urgensi"] == "periksa dokter"


def test_gejala_majemuk_periksa_dokter():
    assert triase(["batuk", "pilek", "pusing"])["urgensi"] == "periksa dokter"


def test_gejala_ringan_periksa_perawat():
    assert triase(["batuk"])["urgensi"] == "periksa perawat"


def test_tanpa_gejala_rawat_mandiri():
    assert triase([])["urgensi"] == "rawat mandiri"


def test_disclaimer_selalu_ada():
    assert triase(["batuk"])["disclaimer"] == DISCLAIMER
