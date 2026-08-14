"""Harness evaluasi kuantitatif untuk detektor wabah.

Menjawab pertanyaan ilmiah: **seberapa baik detektor menemukan wabah nyata,
dan seberapa sering ia salah alarm?**

Cara kerja:

1. Bangkitkan dataset sintetis berlabel - kita mengetahui persis hari & blok
   mana yang benar-benar wabah (ground truth), karena kita yang menyuntikkannya.
2. Jalankan tiap metode deteksi (``zscore``, ``poisson``, ``threshold``) pada
   setiap hari evaluasi.
3. Bandingkan prediksi dengan ground truth -> TP / FP / FN / TN.
4. Laporkan Precision, Recall, F1, dan akurasi, dirata-ratakan atas beberapa
   dataset (seed berbeda) agar hasil tidak bergantung satu sampel.

Unit evaluasi = pasangan (tanggal, blok). Evaluasi dibatasi pada dimensi
``blok`` dan gejala wabah agar penghitungan tidak ganda antar dimensi.

Jalankan: ``python evaluasi.py``
"""
import random
from datetime import date, timedelta

from surveilans import METODE_TERSEDIA, deteksi_anomali

# Parameter detektor yang dipakai selama evaluasi (sama untuk semua metode
# agar perbandingan adil).
CONFIG_EVALUASI = {
    "baseline_hari": 14,
    "z_ambang": 2.0,
    "c_min": 3,
    "p_ambang": 0.05,
    "threshold_kasus": 4,
}

BLOK = ["A", "B", "C"]
KOMPI = ["1", "2", "3"]
ANGKATAN = ["2022", "2023", "2024"]
GEJALA_LATAR = ["batuk", "pilek", "pusing", "mual", "diare", "nyeri tenggorokan"]
GEJALA_WABAH = "demam"

# Ukuran dataset default. Parameter dipilih agar tugas deteksi REALISTIS,
# bukan sepele: gejala wabah juga muncul sebagai keluhan sehari-hari, dan ada
# "hari ramai" (mis. sehabis kegiatan lapangan) yang menaikkan kunjungan tanpa
# benar-benar wabah - inilah sumber alarm palsu yang harus dibedakan detektor.
N_HARI = 60               # total hari yang dibangkitkan
KASUS_LATAR_HARIAN = 6    # rata-rata kunjungan latar per hari (seluruh blok)
PROB_DEMAM_LATAR = 0.35   # demam juga keluhan biasa, bukan hanya tanda wabah
PELUANG_WABAH = 0.10      # peluang satu (hari, blok) menjadi wabah
UKURAN_WABAH = (3, 9)     # rentang kasus tambahan saat wabah (ada yang kecil)
PELUANG_HARI_RAMAI = 0.12 # peluang satu (hari, blok) ramai tanpa wabah
UKURAN_RAMAI = (5, 9)     # tambahan kunjungan umum pada hari ramai


def hitung_metrik(tp, fp, fn, tn):
    """Hitung Precision, Recall, F1, dan akurasi dari confusion matrix.

    Precision dan Recall didefinisikan 0.0 (bukan error) saat penyebut nol,
    konvensi umum agar perbandingan metode tetap bisa dilakukan.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    akurasi = (tp + tn) / total if total else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "akurasi": akurasi,
    }


def buat_dataset_berlabel(seed=0, n_hari=N_HARI, peluang_wabah=PELUANG_WABAH):
    """Bangkitkan (kunjungan, label_wabah) sintetis.

    ``label_wabah`` adalah himpunan pasangan ``(tanggal, blok)`` yang benar-benar
    merupakan wabah - inilah ground truth yang dipakai menilai detektor.
    """
    rng = random.Random(seed)
    hari_akhir = date(2026, 8, 14)  # tanggal tetap agar dataset reprodusibel
    kunjungan = []
    label = set()
    _id = 0

    def _catat(tanggal, blok, gejala):
        nonlocal _id
        _id += 1
        kunjungan.append({
            "id": _id, "tanggal": tanggal, "blok": blok,
            "kompi": rng.choice(KOMPI), "angkatan": rng.choice(ANGKATAN),
            "gejala": [gejala],
        })

    def _gejala_biasa():
        """Keluhan sehari-hari; demam termasuk di dalamnya."""
        if rng.random() < PROB_DEMAM_LATAR:
            return GEJALA_WABAH
        return rng.choice(GEJALA_LATAR)

    for d in range(n_hari - 1, -1, -1):
        tanggal = (hari_akhir - timedelta(days=d)).isoformat()

        # Kunjungan latar - gejala campuran (termasuk demam), blok acak.
        for _ in range(max(0, int(rng.gauss(KASUS_LATAR_HARIAN, 2.0)))):
            _catat(tanggal, rng.choice(BLOK), _gejala_biasa())

        # Wabah & hari ramai hanya muncul setelah baseline terisi, agar
        # detektor punya riwayat pembanding yang wajar.
        if d >= n_hari - CONFIG_EVALUASI["baseline_hari"]:
            continue

        for blok in BLOK:
            undian = rng.random()
            if undian < peluang_wabah:
                # Wabah sesungguhnya: lonjakan gejala spesifik.
                for _ in range(rng.randint(*UKURAN_WABAH)):
                    _catat(tanggal, blok, GEJALA_WABAH)
                label.add((tanggal, blok))
            elif undian < peluang_wabah + PELUANG_HARI_RAMAI:
                # Hari ramai TANPA wabah (mis. sehabis kegiatan lapangan):
                # kunjungan naik dengan komposisi gejala normal. Detektor yang
                # baik seharusnya TIDAK menandai ini.
                for _ in range(rng.randint(*UKURAN_RAMAI)):
                    _catat(tanggal, blok, _gejala_biasa())

    return kunjungan, label


def _hari_evaluasi(kunjungan, baseline_hari):
    """Hari-hari yang layak dinilai (baseline sudah cukup panjang)."""
    semua = sorted({k["tanggal"] for k in kunjungan})
    return semua[baseline_hari:]


def evaluasi_metode(kunjungan, label_wabah, metode, config=None):
    """Nilai satu metode terhadap ground truth; kembalikan metrik."""
    if metode not in METODE_TERSEDIA:
        raise ValueError(
            f"Metode '{metode}' tidak dikenal. Pilih salah satu: {METODE_TERSEDIA}"
        )
    config = config or CONFIG_EVALUASI
    tp = fp = fn = tn = 0

    for tanggal in _hari_evaluasi(kunjungan, config["baseline_hari"]):
        alerts = deteksi_anomali(kunjungan, config, tanggal=tanggal,
                                 dimensi=["blok"], metode=metode)
        diprediksi = {a["grup"] for a in alerts if a["gejala"] == GEJALA_WABAH}
        for blok in BLOK:
            benar_wabah = (tanggal, blok) in label_wabah
            ditandai = blok in diprediksi
            if ditandai and benar_wabah:
                tp += 1
            elif ditandai and not benar_wabah:
                fp += 1
            elif not ditandai and benar_wabah:
                fn += 1
            else:
                tn += 1

    return hitung_metrik(tp, fp, fn, tn)


def _rata_metrik(daftar):
    """Rata-ratakan metrik proporsi, jumlahkan hitungan mentah."""
    n = len(daftar) or 1
    rata = {k: sum(d[k] for d in daftar) / n
            for k in ("precision", "recall", "f1", "akurasi")}
    rata.update({k: sum(d[k] for d in daftar) for k in ("tp", "fp", "fn", "tn")})
    rata["n_dataset"] = len(daftar)
    return rata


def evaluasi_rata(metode, config=None, n_dataset=5):
    """Metrik satu metode, dirata-ratakan atas ``n_dataset`` seed berbeda."""
    daftar = []
    for seed in range(n_dataset):
        kunjungan, label = buat_dataset_berlabel(seed=seed)
        daftar.append(evaluasi_metode(kunjungan, label, metode, config))
    return _rata_metrik(daftar)


def bandingkan_metode(n_dataset=5, config=None):
    """Rata-ratakan metrik tiap metode atas ``n_dataset`` seed berbeda."""
    return {m: evaluasi_rata(m, config, n_dataset) for m in METODE_TERSEDIA}


# Parameter utama tiap metode beserta kandidat nilai untuk penalaan.
PARAM_SWEEP = {
    "zscore": ("z_ambang", [1.0, 1.5, 2.0, 2.5, 3.0]),
    "robust": ("z_ambang", [1.0, 2.0, 3.0, 4.0, 5.0]),
    "poisson": ("p_ambang", [0.2, 0.1, 0.05, 0.01, 0.001]),
    "threshold": ("threshold_kasus", [3, 4, 5, 6, 7]),
}


def sweep_parameter(metode, n_dataset=3):
    """Uji beberapa nilai parameter; kembalikan metrik untuk tiap nilai."""
    if metode not in PARAM_SWEEP:
        raise ValueError(f"Tidak ada parameter sweep untuk metode '{metode}'")
    nama, kandidat = PARAM_SWEEP[metode]
    hasil = []
    for nilai in kandidat:
        config = dict(CONFIG_EVALUASI)
        config[nama] = nilai
        metrik = evaluasi_rata(metode, config, n_dataset)
        metrik.update({"parameter": nama, "nilai": nilai})
        hasil.append(metrik)
    return hasil


def cari_konfigurasi_terbaik(n_dataset=3):
    """Pilih nilai parameter dengan F1 tertinggi untuk tiap metode."""
    return {m: max(sweep_parameter(m, n_dataset), key=lambda x: x["f1"])
            for m in PARAM_SWEEP}


NAMA_METODE = {
    "zscore": "z-score (rata-rata + simpangan baku)",
    "robust": "z-score robust (median + MAD)",
    "poisson": "Uji Poisson satu sisi",
    "threshold": "Ambang kasus tetap (baseline naif)",
}


def format_tabel(hasil):
    """Render hasil perbandingan sebagai tabel Markdown."""
    baris = [
        "| Metode | Precision | Recall | F1 | Akurasi | TP | FP | FN |",
        "|--------|-----------|--------|----|---------|----|----|----|",
    ]
    for metode, m in sorted(hasil.items(), key=lambda x: x[1]["f1"], reverse=True):
        baris.append(
            f"| `{metode}` - {NAMA_METODE.get(metode, metode)} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['akurasi']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |"
        )
    return "\n".join(baris)


def format_tabel_tuning(terbaik):
    """Render hasil penalaan parameter sebagai tabel Markdown."""
    baris = [
        "| Metode | Parameter | Nilai terbaik | Precision | Recall | F1 |",
        "|--------|-----------|---------------|-----------|--------|----|",
    ]
    for metode, m in sorted(terbaik.items(), key=lambda x: x[1]["f1"], reverse=True):
        baris.append(
            f"| `{metode}` | `{m['parameter']}` | {m['nilai']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    return "\n".join(baris)


def main(n_dataset=5, n_dataset_tuning=3, tulis_laporan=True):
    default = bandingkan_metode(n_dataset=n_dataset)
    terbaik = cari_konfigurasi_terbaik(n_dataset=n_dataset_tuning)
    juara = max(terbaik.items(), key=lambda x: x[1]["f1"])

    bagian = [
        "# Hasil Evaluasi Detektor Wabah",
        "",
        "Dihasilkan otomatis oleh `evaluasi.py`. Seluruh angka berasal dari "
        "dataset **sintetis berlabel** - kita mengetahui persis hari & blok mana "
        "yang benar-benar wabah, sehingga Precision/Recall/F1 dapat dihitung.",
        "",
        f"- Dataset uji: **{n_dataset}** seed berbeda, masing-masing {N_HARI} hari",
        f"- Unit evaluasi: pasangan **(tanggal, blok)**; gejala wabah: `{GEJALA_WABAH}`",
        "- Tantangan yang sengaja dibuat: gejala wabah juga muncul sebagai keluhan "
        "harian, dan ada **hari ramai tanpa wabah** sebagai sumber alarm palsu.",
        "",
        "## 1. Konfigurasi default",
        "",
        format_tabel(default),
        "",
        f"## 2. Setelah penalaan parameter ({n_dataset_tuning} dataset)",
        "",
        format_tabel_tuning(terbaik),
        "",
        f"**F1 tertinggi setelah penalaan: `{juara[0]}` "
        f"({juara[1]['f1']:.3f}) pada {juara[1]['parameter']}={juara[1]['nilai']}.**",
        "",
        "## Catatan kejujuran",
        "",
        "Angka di atas berlaku pada data sintetis dengan asumsi yang dijelaskan di "
        "`evaluasi.py`, bukan pada data poliklinik nyata. Tujuannya membandingkan "
        "metode secara adil pada kondisi yang sama, bukan mengklaim akurasi lapangan.",
        "",
    ]
    laporan = "\n".join(bagian)

    print(laporan)
    if tulis_laporan:
        with open("EVALUASI.md", "w", encoding="utf-8") as f:
            f.write(laporan)
        print("Laporan ditulis ke EVALUASI.md")
    return {"default": default, "terbaik": terbaik}


if __name__ == "__main__":
    main()
