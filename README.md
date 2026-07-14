# Asisten Poliklinik Taruna (Agentic AI)

Asisten AI lokal untuk petugas poliklinik: mendigitalkan pendataan pasien &
surat keterangan sakit lewat chat, plus deteksi dini klaster/wabah asrama.

> Semua data dummy/sintetis. Bukan alat diagnosis — keputusan medis ada pada
> tenaga kesehatan.

## Prasyarat
- Python 3.10+
- [Ollama](https://ollama.com) dengan model `qwen3:8b` (`ollama pull qwen3:8b`).
  Alternatif lebih ringan/cepat diunduh: `qwen3:4b`.

## Setup
```bash
pip install -r requirements.txt
python seed_data.py        # buat data dummy master + kunjungan
streamlit run app.py       # jalankan aplikasi
```

## Cara pakai (ketik di chat)
- `Catat Budi Pratama demam batuk suhu 38.5` — mencatat kunjungan
- `Ada potensi wabah minggu ini?` — cek klaster/anomali
- `Buatkan surat sakit Budi Pratama istirahat 2 hari` — surat keterangan
- `Rekap hari ini` — ringkasan harian
- `Riwayat berobat Budi Pratama` — riwayat pasien

## Arsitektur
Chat (Streamlit) -> agent (Ollama qwen3:8b, tool-calling) -> tool Python
(triase / surat / surveilans / storage) -> data JSON.

## Inti ML — deteksi anomali
Deteksi lonjakan gejala per blok/kompi/angkatan dengan baseline rata-rata
bergerak + z-score (`z >= 2` dan `c_t >= 3`). Lihat `surveilans.py`.

## Uji
```bash
pytest -v
```

## Catatan model
Jika balasan tidak memanggil tool, pastikan Ollama versi terbaru dan model
mendukung tool-calling (`qwen3:8b`/`qwen3:4b`). Ganti nama model di
`agent.py` (variabel `MODEL`) bila memakai varian lain.
