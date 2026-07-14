import random
from datetime import date, timedelta

import storage

NAMA_DEPAN = ["Budi", "Andi", "Cahya", "Dewi", "Eka", "Fajar", "Gilang",
              "Hana", "Indra", "Joko", "Kirana", "Lukman", "Maya", "Nur",
              "Oki", "Putri", "Rizki", "Sari", "Tono", "Umar"]
NAMA_BELAKANG = ["Pratama", "Wijaya", "Santoso", "Kusuma", "Halim",
                 "Nugroho", "Saputra", "Utami", "Permana", "Lestari"]
BLOK = ["A", "B", "C"]
KOMPI = ["1", "2", "3"]
ANGKATAN = ["2022", "2023", "2024"]
GEJALA_UMUM = ["demam", "batuk", "pilek", "pusing", "mual", "diare",
               "nyeri tenggorokan"]


def generate_master(n=60, seed=42):
    rng = random.Random(seed)
    master = []
    for i in range(n):
        nama = f"{rng.choice(NAMA_DEPAN)} {rng.choice(NAMA_BELAKANG)}"
        master.append({
            "no_taruna": f"{rng.choice(ANGKATAN)}{1000 + i:04d}",
            "nama": nama,
            "blok": rng.choice(BLOK),
            "kompi": rng.choice(KOMPI),
            "angkatan": rng.choice(ANGKATAN),
        })
    return master


def generate_kunjungan(master, hari=21, rata2_per_hari=4, spike=True, seed=42):
    rng = random.Random(seed)
    hari_ini = date.today()
    kunjungan = []
    _id = 0
    for d in range(hari - 1, -1, -1):
        tanggal = (hari_ini - timedelta(days=d)).isoformat()
        jumlah = max(0, int(rng.gauss(rata2_per_hari, 1.5)))
        for _ in range(jumlah):
            t = rng.choice(master)
            gejala = rng.sample(GEJALA_UMUM, k=rng.randint(1, 2))
            suhu = round(rng.uniform(36.5, 38.5), 1)
            _id += 1
            kunjungan.append({
                "id": _id, "tanggal": tanggal,
                "no_taruna": t["no_taruna"], "nama": t["nama"],
                "blok": t["blok"], "kompi": t["kompi"], "angkatan": t["angkatan"],
                "gejala": gejala, "suhu": suhu,
                "urgensi": "periksa perawat", "catatan": "",
            })
    if spike:
        tanggal = hari_ini.isoformat()
        kandidat = [t for t in master if t["blok"] == "A"] or master
        for _ in range(6):
            t = rng.choice(kandidat)
            _id += 1
            kunjungan.append({
                "id": _id, "tanggal": tanggal,
                "no_taruna": t["no_taruna"], "nama": t["nama"],
                "blok": "A", "kompi": t["kompi"], "angkatan": t["angkatan"],
                "gejala": ["demam"], "suhu": round(rng.uniform(38.0, 39.5), 1),
                "urgensi": "periksa dokter", "catatan": "cluster demam",
            })
    return kunjungan


def main():
    master = generate_master()
    storage.save_master(master)
    kunjungan = generate_kunjungan(master)
    storage.save_kunjungan(kunjungan)
    print(f"Tersimpan: {len(master)} taruna, {len(kunjungan)} kunjungan.")


if __name__ == "__main__":
    main()
