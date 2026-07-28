# 🎤 Skrip Demo & Poin Bicara

Panduan lengkap untuk memaparkan **Asisten Poliklinik Taruna** ke dosen.
Berisi: checklist persiapan, alur paparan, skrip demo langkah-demi-langkah
(apa yang diketik + hasil + poin bicara), dan antisipasi bila demo bermasalah.

---

## ✅ Checklist Persiapan (lakukan 10 menit sebelum paparan)

- [ ] Ollama berjalan → cek `ollama list` memuat `qwen3:4b` (ringan, muat di GPU 6 GB)
- [ ] **Re-seed data hari ini** → `python seed_data.py`
      (penting: lonjakan wabah menempel ke tanggal hari-H, jadi "Ada wabah?" langsung menyala)
- [ ] Jalankan app → `streamlit run app.py` → buka **http://localhost:8501**
- [ ] Coba 1 perintah dulu (mis. "Rekap hari ini") untuk "memanaskan" model
- [ ] Buka terminal kedua, siapkan `pytest -v` (untuk tunjukkan bukti pengujian)
- [ ] Perbesar font browser & terminal agar terbaca dari jauh

> ⏱️ **Catatan penting:** model lokal butuh beberapa detik (~5–20 dtk untuk
> qwen3:4b) per jawaban. Ini **normal**. Manfaatkan jeda itu untuk menjelaskan apa
> yang sedang terjadi ("sekarang LLM sedang memutuskan tool mana yang dipanggil...").
>
> 💾 **Model & GPU:** GPU 6 GB VRAM cocok dengan **qwen3:4b**. `qwen3:8b` bisa
> kehabisan VRAM & membuat Ollama crash saat browser terbuka — hindari untuk demo.

---

## 🗺️ Alur Paparan (± 10–12 menit)

| Bagian | Durasi | Isi |
|--------|--------|-----|
| 1. Pembukaan & masalah | 1–2 mnt | Kenapa proyek ini dibuat |
| 2. Konsep Agentic AI | 1–2 mnt | Apa yang membedakan dari chatbot biasa |
| 3. **Demo langsung** | 5–6 mnt | Tunjukkan aplikasi bekerja |
| 4. Inti Machine Learning | 2 mnt | Metode deteksi wabah (z-score) |
| 5. Penutup | 1 mnt | Keamanan, keterbatasan, rencana |

---

## 🗣️ Bagian 1 — Pembukaan & Masalah

> "Selamat pagi/siang. Saya akan memaparkan produk **Agentic AI** bernama
> *Asisten Poliklinik Taruna*. Masalah yang saya angkat: di poliklinik kampus,
> **pendataan pasien dan pembuatan surat sakit masih manual**, dan **tidak ada
> sistem yang memantau potensi wabah** padahal taruna tinggal berdekatan di asrama.
> Produk ini menyelesaikan ketiganya lewat satu asisten chat."

**Poin kunci:** masalah nyata → solusi konkret.

---

## 🗣️ Bagian 2 — Konsep Agentic AI (pembeda dari chatbot)

> "Yang membuat ini *agentic*, bukan sekadar chatbot: **LLM tidak menghitung atau
> menyimpan data sendiri**. LLM hanya bertugas **memilih tool mana yang dipanggil
> dengan argumen apa**. Yang benar-benar bekerja adalah fungsi Python di baliknya.
> Jadi alurnya: *menalar → pilih tool → bertindak → rangkai jawaban*."

**Poin kunci:** LLM = otak pengambil keputusan; tool = tangan yang bekerja.
(Boleh tunjuk diagram arsitektur di README.)

---

## 🎬 Bagian 3 — Demo Langsung (SKRIP)

**Framing pembuka:**
> "Bayangkan saya petugas poliklinik. Hari ini beberapa taruna datang berobat.
> Saya cukup mengetik dengan bahasa biasa."

### Langkah 1 — Mencatat pasien (fitur inti: anti-manual)

🎬 **KETIK:**
```
Catat Dewi Pratama, demam dan batuk, suhu 38.6
```
✅ **HASIL:** konfirmasi tercatat + lokasi (Blok/Kompi/Angkatan) **terisi otomatis**
+ urgensi "periksa dokter" + disclaimer.

🗣️ **JELASKAN:**
> "Perhatikan: saya tidak mengisi form. LLM memilih tool `catat_kunjungan`, dan
> **lokasi taruna terisi otomatis** dari data induk — ini yang menggantikan
> pendataan manual. Urgensi juga dinilai otomatis oleh aturan triase."

### Langkah 2 — Triase (penilaian urgensi + rambu keamanan)

🎬 **KETIK:**
```
Kalau ada taruna sesak napas dan nyeri dada, seberapa urgen?
```
✅ **HASIL:** Urgensi **Segera 🚨** + catatan "**bukan diagnosis**".

🗣️ **JELASKAN:**
> "Sistem menilai urgensi untuk prioritas antrean, TAPI selalu menegaskan ini
> **bukan diagnosis** — keputusan medis tetap di tenaga kesehatan. Ini pertimbangan
> etika/keamanan yang saya sengaja bangun."

### Langkah 3 — Surat keterangan sakit (anti-manual #2)

🎬 **KETIK:**
```
Buatkan surat sakit Dewi Pratama istirahat 2 hari
```
✅ **HASIL:** draft surat keterangan sakit lengkap, siap cetak.

🗣️ **JELASKAN:**
> "Surat yang tadinya diketik manual, kini tergenerate otomatis dari data kunjungan."

### Langkah 4 — Rekap harian

🎬 **KETIK:**
```
Rekap hari ini
```
✅ **HASIL:** jumlah kunjungan, gejala terbanyak, sebaran per blok.

🗣️ **JELASKAN:**
> "Karena datanya sudah digital, rekap harian instan — tidak perlu hitung manual."

### Langkah 5 — ⭐ PUNCAK: Deteksi Wabah (inti ML)

🎬 **KETIK:**
```
Ada potensi wabah minggu ini?
```
✅ **HASIL:** daftar klaster, mis. **⚠️ Blok A: 7 kasus demam (z=9.59)**.
Sambil itu, **tunjuk panel dashboard kanan** — grafik batang + kotak peringatan.

🗣️ **JELASKAN:**
> "Inilah nilai tambah cerdasnya. Sistem membandingkan jumlah kasus hari ini
> dengan pola normal (baseline) tiap blok/kompi/angkatan. Kalau lonjakannya
> jauh di atas normal — diukur pakai **z-score** — sistem memberi peringatan dini.
> z=9.59 artinya lonjakan ini **sangat** tidak normal."

### Langkah 6 — Riwayat pasien (opsional bila waktu cukup)

🎬 **KETIK:**
```
Riwayat berobat Dewi Pratama
```
✅ **HASIL:** daftar kunjungan taruna tersebut.

---

## 🗣️ Bagian 4 — Inti Machine Learning (di depan dashboard)

> "Metode deteksinya adalah **surveilans sindromik dengan z-score**. Untuk tiap
> grup lokasi dan gejala, saya hitung rata-rata (μ) dan simpangan baku (σ) dari
> **14 hari** ke belakang sebagai baseline. Lalu:
> **z = (kasus hari ini − μ) / σ**.
> Kalau z ≥ 2 DAN kasus ≥ 3, ditandai sebagai potensi klaster. Metode ini saya
> pilih karena **mudah dijelaskan** (explainable) dan bekerja dengan data terbatas.
> Saya juga menyediakan uji Poisson sebagai pembanding."

**Poin kunci:** ini konten Machine Learning-nya — belajar pola normal, tandai anomali.

---

## 🗣️ Bagian 5 — Penutup

> "Sebagai penutup: seluruh data di sini **100% sintetis**, karena memakai data
> taruna asli akan menimbulkan isu privasi — relevan dengan latar kampus siber.
> Sistem berjalan **sepenuhnya lokal** tanpa cloud. Keterbatasannya: model lokal
> kecil kadang perlu prompt tegas. Rencana lanjut: impor data via CSV dan
> notifikasi otomatis saat wabah terdeteksi. Kode diuji dengan **28 unit test**
> dan dikembangkan secara bertahap (TDD). Terima kasih."

---

## 🧰 Bukti Pendukung (bila diminta)

Tunjukkan di terminal kedua:
```bash
pytest -v          # 28 test lulus
git log --oneline  # riwayat pengembangan bertahap (13 commit)
```

---

## 🚑 Antisipasi Masalah Saat Demo

| Gejala | Penyebab & Solusi Cepat |
|--------|-------------------------|
| Jawaban lama muncul | Normal untuk model lokal (~5–20 dtk). Jelaskan sambil menunggu. |
| Error CUDA / Ollama crash | VRAM penuh (jangan pakai qwen3:8b di GPU 6 GB). Pakai `qwen3:4b`. |
| "Ada wabah?" bilang tidak ada | Belum re-seed hari ini → jalankan `python seed_data.py`, refresh app. |
| LLM tidak memanggil tool / salah jawab | Ulangi dengan kalimat lebih spesifik & tegas (sebut aksinya jelas). |
| App error / tak mau jalan | Tunjukkan `pytest -v` (28 lulus) sebagai bukti logika bekerja, lalu restart app. |
| Ollama tak merespons | Cek Ollama berjalan: `ollama list`; jalankan ulang bila perlu. |

---

## 💡 Tips Presentasi

- **Ceritakan sebagai skenario**, bukan daftar fitur ("bayangkan saya petugas...").
- Saat menunggu jawaban model, **jangan diam** — jelaskan proses agentic-nya.
- **Puncak demo = deteksi wabah.** Beri jeda dramatis di situ, tunjuk dashboard.
- Jujur soal keterbatasan → menunjukkan pemahaman kritis (dosen menghargai ini).
