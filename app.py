import pandas as pd
import streamlit as st

import agent
import storage
from surveilans import deteksi_anomali

st.set_page_config(page_title="Asisten Poliklinik Taruna", layout="wide")
st.title("🏥 Asisten Poliklinik Taruna")
st.caption("Asisten AI untuk petugas: pencatatan, surat, dan deteksi dini wabah. "
           "Bukan alat diagnosis.")

col_chat, col_dash = st.columns([3, 2])

with col_chat:
    st.subheader("💬 Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Contoh: Catat Budi demam batuk suhu 38.5"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Memproses..."):
                balasan, hist = agent.chat(prompt, st.session_state.history)
            st.session_state.history = [
                m for m in hist
                if m.get("role") in ("user", "assistant", "tool")
            ][-10:]
            st.markdown(balasan)
        st.session_state.messages.append({"role": "assistant", "content": balasan})

with col_dash:
    st.subheader("📊 Surveilans")
    kunjungan = storage.load_kunjungan()
    if not kunjungan:
        st.info("Belum ada data. Jalankan `python seed_data.py` dulu.")
    else:
        df = pd.DataFrame(kunjungan).explode("gejala")
        pivot = (df.groupby(["tanggal", "blok"]).size()
                 .reset_index(name="kasus")
                 .pivot(index="tanggal", columns="blok", values="kasus")
                 .fillna(0))
        st.bar_chart(pivot)

        config = storage.load_config()
        metode = config.get("metode", "poisson")
        alerts = deteksi_anomali(kunjungan, config, metode=metode)
        st.caption(f"Metode deteksi: `{metode}` (lihat EVALUASI.md untuk "
                   "perbandingan kuantitatif antar metode)")
        if alerts:
            for a in alerts:
                skor = (f"p={a['p_value']:.4f}" if "p_value" in a
                        else f"z={a['z']}" if "z" in a
                        else f"ambang={a.get('ambang')}")
                st.error(f"⚠️ {a['dimensi'].capitalize()} {a['grup']}: "
                         f"{a['c_t']} kasus '{a['gejala']}' "
                         f"(baseline {a['mu']}±{a['sigma']}, {skor})")
        else:
            st.success("Tidak ada anomali terdeteksi.")
