DISCLAIMER = (
    "Catatan: ini penilaian urgensi awal, BUKAN diagnosis. "
    "Keputusan medis ada pada perawat/dokter."
)

GEJALA_DARURAT = {
    "sesak napas",
    "nyeri dada",
    "pingsan",
    "kejang",
    "perdarahan",
    "penurunan kesadaran",
}


def triase(gejala, suhu=None):
    g = {str(x).strip().lower() for x in gejala if str(x).strip()}
    alasan = []

    darurat = g & GEJALA_DARURAT
    if darurat:
        urgensi = "segera"
        alasan.append("ada gejala kegawatan: " + ", ".join(sorted(darurat)))
    elif suhu is not None and suhu >= 39.0:
        urgensi = "segera"
        alasan.append(f"suhu sangat tinggi ({suhu} C)")
    elif (suhu is not None and suhu >= 38.0) or len(g) >= 3:
        urgensi = "periksa dokter"
        if suhu is not None and suhu >= 38.0:
            alasan.append(f"demam ({suhu} C)")
        if len(g) >= 3:
            alasan.append("gejala majemuk")
    elif len(g) >= 1:
        urgensi = "periksa perawat"
        alasan.append("gejala ringan")
    else:
        urgensi = "rawat mandiri"
        alasan.append("tidak ada gejala berarti")

    return {"urgensi": urgensi, "alasan": "; ".join(alasan), "disclaimer": DISCLAIMER}
