from datetime import date


def buat_surat(nama, lama_istirahat, keterangan="", tanggal=None, petugas="Petugas Poliklinik"):
    tanggal = tanggal or date.today().isoformat()
    baris_ket = f"\nKeterangan: {keterangan}" if keterangan else ""
    return (
        "SURAT KETERANGAN SAKIT\n"
        "Poliklinik Taruna\n\n"
        "Yang bertanda tangan di bawah ini menerangkan bahwa:\n"
        f"  Nama : {nama}\n\n"
        f"Berdasarkan pemeriksaan pada {tanggal}, yang bersangkutan dianjurkan\n"
        f"ISTIRAHAT selama {lama_istirahat} hari terhitung sejak tanggal tersebut."
        f"{baris_ket}\n\n"
        f"{tanggal}\n"
        "Mengetahui,\n"
        f"{petugas}\n"
    )
