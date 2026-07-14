import ollama

import tools as T

MODEL = "qwen3:8b"
SYSTEM_PROMPT = (
    "Anda adalah asisten poliklinik taruna untuk PETUGAS klinik. "
    "Gunakan tool yang tersedia untuk: mencatat kunjungan, menilai urgensi (triase), "
    "membuat surat keterangan sakit, memeriksa anomali/klaster gejala (potensi wabah), "
    "membuat rekap harian, dan menampilkan riwayat pasien. "
    "JANGAN mendiagnosis penyakit atau memberi resep obat; itu wewenang tenaga kesehatan. "
    "Jawab ringkas, jelas, dan dalam Bahasa Indonesia."
)


def chat(user_message, history=None, max_iter=5):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += (history or [])
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_iter):
        resp = ollama.chat(model=MODEL, messages=messages, tools=T.TOOLS)
        msg = resp.message
        tool_calls = getattr(msg, "tool_calls", None)
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
        })
        if not tool_calls:
            return msg.content or "", messages
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

    return "Maaf, terlalu banyak langkah. Coba perjelas permintaan.", messages
