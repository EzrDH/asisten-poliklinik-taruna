import ollama

import tools as T

MODEL = "qwen3:8b"
SYSTEM_PROMPT = (
    "Anda adalah asisten poliklinik taruna untuk PETUGAS klinik. "
    "Gunakan tool yang tersedia untuk: mencatat kunjungan, menilai urgensi (triase), "
    "membuat surat keterangan sakit, memeriksa anomali/klaster gejala (potensi wabah), "
    "membuat rekap harian, dan menampilkan riwayat pasien. "
    "JANGAN mendiagnosis penyakit atau memberi resep obat; itu wewenang tenaga kesehatan. "
    "Untuk parameter tanggal/periode yang opsional, JANGAN mengarang nilainya; "
    "biarkan kosong kecuali pengguna menyebut tanggal tertentu. "
    "Jawab ringkas, jelas, dan dalam Bahasa Indonesia."
)


def chat(user_message, history=None, max_iter=5):
    history = history or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    reply = "Maaf, terlalu banyak langkah. Coba perjelas permintaan."
    for _ in range(max_iter):
        resp = ollama.chat(model=MODEL, messages=messages, tools=T.TOOLS)
        msg = resp.message
        # Simpan objek pesan asisten apa adanya agar tool_calls tetap tertaut
        # ke hasil tool -> model membaca output tool dengan benar.
        messages.append(msg)
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            reply = msg.content or ""
            break
        for call in tool_calls:
            nama = call.function.name
            args = call.function.arguments or {}
            fungsi = T.TOOL_MAP.get(nama)
            if fungsi is None:
                hasil = f"Tool '{nama}' tidak dikenal."
            else:
                try:
                    hasil = fungsi(**args)
                except Exception as e:  # noqa: BLE001
                    hasil = f"Error menjalankan {nama}: {e}"
            messages.append({"role": "tool", "name": nama, "content": str(hasil)})

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return reply, new_history
