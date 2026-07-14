from collections import Counter
from datetime import date

import storage
from triase import triase
from surat import buat_surat
from surveilans import deteksi_anomali


def _normalize_gejala(gejala):
    if isinstance(gejala, str):
        return [g.strip().lower() for g in gejala.split(",") if g.strip()]
    return [str(g).strip().lower() for g in gejala if str(g).strip()]


def cari_taruna(nama_atau_no):
    q = str(nama_atau_no).strip().lower()
    for t in storage.load_master():
        if (t["nama"].lower() == q or t["no_taruna"].lower() == q
                or q in t["nama"].lower()):
            return t
    return None


def catat_kunjungan(nama, gejala, suhu=None, catatan=""):
    t = cari_taruna(nama)
    if not t:
        return (f"Taruna '{nama}' tidak ditemukan di data master. "
                "Mohon periksa nama/nomor taruna.")
    g = _normalize_gejala(gejala)
    if not g:
        return "Gejala kosong. Sebutkan minimal satu gejala."
    hasil = triase(g, suhu)
    record = {
        "tanggal": date.today().isoformat(),
        "no_taruna": t["no_taruna"], "nama": t["nama"],
        "blok": t["blok"], "kompi": t["kompi"], "angkatan": t["angkatan"],
        "gejala": g, "suhu": suhu, "urgensi": hasil["urgensi"], "catatan": catatan,
    }
    saved = storage.tambah_kunjungan(record)
    suhu_txt = f", suhu {suhu} C" if suhu is not None else ""
    return (f"Tercatat #{saved['id']}: {t['nama']} "
            f"(Blok {t['blok']}/Kompi {t['kompi']}/Angk {t['angkatan']}), "
            f"gejala {', '.join(g)}{suhu_txt}. "
            f"Urgensi: {hasil['urgensi']} ({hasil['alasan']}). {hasil['disclaimer']}")


def triase_tool(gejala, suhu=None):
    g = _normalize_gejala(gejala)
    hasil = triase(g, suhu)
    return f"Urgensi: {hasil['urgensi']} ({hasil['alasan']}). {hasil['disclaimer']}"


def buat_surat_sakit(nama, lama_istirahat, keterangan=""):
    t = cari_taruna(nama)
    nama_lengkap = t["nama"] if t else nama
    return buat_surat(nama_lengkap, int(lama_istirahat), keterangan=keterangan)


def cek_anomali(periode=None, dimensi=None):
    kunjungan = storage.load_kunjungan()
    config = storage.load_config()
    if isinstance(dimensi, str):
        dimensi = [dimensi]
    alerts = deteksi_anomali(kunjungan, config, dimensi=dimensi)
    if not alerts:
        return "Tidak terdeteksi anomali/klaster gejala di atas ambang saat ini."
    baris = [
        f"⚠️ {a['dimensi'].capitalize()} {a['grup']}: {a['c_t']} kasus "
        f"'{a['gejala']}' (baseline {a['mu']}±{a['sigma']}, z={a['z']}) "
        f"pada {a['tanggal']}"
        for a in alerts
    ]
    return "Potensi klaster terdeteksi:\n" + "\n".join(baris)


def rekap_harian(tanggal=None):
    kunjungan = storage.load_kunjungan()
    tanggal = tanggal or date.today().isoformat()
    hari = [k for k in kunjungan if k["tanggal"] == tanggal]
    if not hari:
        return f"Tidak ada kunjungan tercatat pada {tanggal}."
    gejala_count = Counter(g for k in hari for g in k["gejala"])
    blok_count = Counter(k["blok"] for k in hari)
    top_gejala = ", ".join(f"{g} ({n})" for g, n in gejala_count.most_common(3))
    sebaran = ", ".join(f"Blok {b}: {n}" for b, n in blok_count.most_common())
    return (f"Rekap {tanggal}: {len(hari)} kunjungan. "
            f"Gejala terbanyak: {top_gejala}. Sebaran: {sebaran}.")


def riwayat_pasien(nama):
    t = cari_taruna(nama)
    kunjungan = storage.load_kunjungan()
    if t:
        rows = [k for k in kunjungan if k["no_taruna"] == t["no_taruna"]]
    else:
        rows = [k for k in kunjungan if str(nama).lower() in k["nama"].lower()]
    if not rows:
        return f"Tidak ada riwayat kunjungan untuk '{nama}'."
    rows.sort(key=lambda k: k["tanggal"])
    baris = [f"- {k['tanggal']}: {', '.join(k['gejala'])} "
             f"(urgensi {k['urgensi']})" for k in rows]
    return f"Riwayat {rows[0]['nama']} ({len(rows)} kunjungan):\n" + "\n".join(baris)


TOOL_MAP = {
    "catat_kunjungan": catat_kunjungan,
    "triase": triase_tool,
    "buat_surat_sakit": buat_surat_sakit,
    "cek_anomali": cek_anomali,
    "rekap_harian": rekap_harian,
    "riwayat_pasien": riwayat_pasien,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "catat_kunjungan",
            "description": "Catat satu kunjungan pasien. Lokasi (blok/kompi/angkatan) terisi otomatis dari data master, dan urgensi dihitung otomatis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nama": {"type": "string", "description": "Nama atau nomor taruna"},
                    "gejala": {"type": "string", "description": "Gejala, pisahkan dengan koma"},
                    "suhu": {"type": "number", "description": "Suhu tubuh dalam Celcius (opsional)"},
                    "catatan": {"type": "string", "description": "Catatan tambahan (opsional)"},
                },
                "required": ["nama", "gejala"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "triase",
            "description": "Nilai tingkat urgensi dari gejala (tanpa mencatat). Bukan diagnosis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gejala": {"type": "string", "description": "Gejala, pisahkan dengan koma"},
                    "suhu": {"type": "number", "description": "Suhu tubuh Celcius (opsional)"},
                },
                "required": ["gejala"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buat_surat_sakit",
            "description": "Buat draft surat keterangan sakit / istirahat untuk seorang taruna.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nama": {"type": "string", "description": "Nama atau nomor taruna"},
                    "lama_istirahat": {"type": "integer", "description": "Jumlah hari istirahat"},
                    "keterangan": {"type": "string", "description": "Keterangan singkat (opsional)"},
                },
                "required": ["nama", "lama_istirahat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cek_anomali",
            "description": "Deteksi lonjakan/klaster gejala per blok/kompi/angkatan untuk peringatan dini wabah.",
            "parameters": {
                "type": "object",
                "properties": {
                    "periode": {"type": "string", "description": "Tanggal (opsional, default hari ini)"},
                    "dimensi": {"type": "string", "description": "blok, kompi, atau angkatan (opsional, default semua)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rekap_harian",
            "description": "Ringkasan kunjungan pada satu hari: jumlah, gejala terbanyak, sebaran blok.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tanggal": {"type": "string", "description": "Tanggal YYYY-MM-DD (opsional, default hari ini)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "riwayat_pasien",
            "description": "Tampilkan riwayat kunjungan seorang taruna.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nama": {"type": "string", "description": "Nama atau nomor taruna"},
                },
                "required": ["nama"],
            },
        },
    },
]
