import streamlit as st
import sqlite3
import joblib
import pandas as pd
import re
from datetime import datetime
import altair as alt

# ========== 1. Load Model & Encoder ==========
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer (1).pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ========== 2. Utility Functions ==========
def tampilkan_logo():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.image("logo_sentimen.png", width=60)

def init_db():
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS status (
            id_status INTEGER PRIMARY KEY AUTOINCREMENT,
            isi_status TEXT,
            label_sentimen TEXT,
            kepercayaan REAL,
            tanggal_status DATE,
            id_user TEXT
        )
    ''')
    conn.commit()
    conn.close()

def simpan_status(id_user, isi_status, label_sentimen, confidence):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute("INSERT INTO status (id_user, isi_status, label_sentimen, kepercayaan, tanggal_status) VALUES (?, ?, ?, ?, ?)",
              (id_user, isi_status, label_sentimen, confidence, datetime.today().date()))
    conn.commit()
    conn.close()

def ambil_riwayat(id_user):
    conn = sqlite3.connect('sentimen.db')
    df = pd.read_sql_query("SELECT * FROM status WHERE id_user = ? ORDER BY id_status DESC", conn, params=(id_user,))
    conn.close()
    return df

def hapus_semua_riwayat(id_user):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute("DELETE FROM status WHERE id_user = ?", (id_user,))
    conn.commit()
    conn.close()

def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    return teks.strip()

def validasi_input(teks, minimal_kata=3):
    return len(teks.strip().split()) >= minimal_kata

def prediksi_sentimen(teks):
    if not validasi_input(teks):
        return "Terlalu pendek", 0
    teks_bersih = bersihkan_teks(teks)
    fitur = tfidf.transform([teks_bersih])
    pred = model.predict(fitur)
    prob = model.predict_proba(fitur).max() * 100
    label = label_encoder.inverse_transform(pred)[0]
    return label.upper(), round(prob, 2)

def rekomendasi_sentimen(label):
    if label == "POSITIF":
        return """
<span style='color:green; font-size:16px;'>☺ <strong>Tetap pertahankan energi positifmu!</strong></span><br>
Lanjutkan membagikan semangat kepada orang lain dan simpan momen positif untuk refleksi di masa mendatang.
"""
    elif label == "NETRAL":
        return """
<span style='color:orange; font-size:16px;'>🔎 <strong>Status kamu cukup netral hari ini.</strong></span><br>
Coba tanyakan pada diri sendiri apa yang sebenarnya kamu rasakan, dan eksplorasi emosi lebih dalam lewat journaling atau refleksi.
"""
    elif label == "NEGATIF":
        return """
<span style='color:red; font-size:16px;'>😟 <strong>Kamu terlihat sedang kurang baik-baik saja.</strong></span><br>
Luangkan waktu untuk istirahat atau melakukan hal-hal yang menenangkan. Jangan ragu untuk bercerita ke orang terpercaya.
"""
    else:
        return ""

# ========== 3. Init ==========
init_db()
if "page" not in st.session_state:
    st.session_state.page = "login"

# ========== 4. Login Page ==========
if st.session_state.page == "login":
    tampilkan_logo()
    st.title("🔐 Masukkan Nama")

    username = st.text_input("Masukkan Nama Pengguna:")
    if username:
        st.session_state.username = username
        st.session_state.page = "home"
        st.rerun()

# ========== 5. Home Page ==========
elif st.session_state.page == "home":
    tampilkan_logo()
    st.title(f"Halo, {st.session_state.username} 👋")
    st.write("Silakan pilih menu:")
    if st.button("➕ Input Status"):
        st.session_state.page = "input"
        st.rerun()
    if st.button("📊 Lihat Riwayat & Hasil"):
        st.session_state.page = "hasil"
        st.rerun()

# ========== 6. Input Status Page ==========
elif st.session_state.page == "input":
    tampilkan_logo()
    st.title("📝 Tulis Status")
    if st.button("← Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()

    status = st.text_area("Apa yang sedang Anda pikirkan hari ini?", height=150)

    if st.button("🔍 Analisis Sekarang"):
        if status.strip():
            label, conf = prediksi_sentimen(status)
            if label == "Terlalu pendek":
                st.warning("⚠️ Status minimal harus terdiri dari 3 kata.")
            else:
                simpan_status(st.session_state.username, status, label, conf)
                st.session_state.hasil_status = status
                st.session_state.hasil_label = label
                st.session_state.hasil_conf = conf
                st.session_state.page = "hasil"
                st.rerun()
        else:
            st.warning("Status tidak boleh kosong.")

# ========== 7. Hasil dan Riwayat ==========
elif st.session_state.page == "hasil":
    tampilkan_logo()
    st.title("📊 Hasil Analisis")

    if st.button("← Kembali ke Input"):
        st.session_state.page = "input"
        st.rerun()

    if "hasil_status" in st.session_state:
        st.subheader("Status Terakhir:")
        st.write(f'💬 **"{st.session_state.hasil_status}"**')
        st.write(f'📌 Sentimen: **{st.session_state.hasil_label}**')
        st.write(f'📈 Kepercayaan: **{st.session_state.hasil_conf}%**')

        rekom = rekomendasi_sentimen(st.session_state.hasil_label)
        if rekom:
            st.markdown("---")
            st.markdown("💡 <h4>Rekomendasi untuk Anda:</h4>", unsafe_allow_html=True)
            st.markdown(rekom, unsafe_allow_html=True)

    st.subheader("🕒 Riwayat Status Anda")
    if st.button("🗑 Hapus Semua Riwayat"):
        hapus_semua_riwayat(st.session_state.username)
        st.success("✅ Riwayat berhasil dihapus.")
        st.rerun()

    df = ambil_riwayat(st.session_state.username)
    if not df.empty:
        df['tanggal_status'] = pd.to_datetime(df['tanggal_status'])
        df_tampil = df.rename(columns={
            'tanggal_status': 'Tanggal',
            'isi_status': 'Status',
            'label_sentimen': 'Label Sentimen',
            'kepercayaan': 'Kepercayaan (%)'
        })[['Tanggal', 'Status', 'Label Sentimen', 'Kepercayaan (%)']]
        st.dataframe(df_tampil, use_container_width=True, hide_index=True)

        # ========== Tren Sentimen Mingguan ==========
        st.markdown("---")
        st.subheader("📈 Tren Sentimen Mingguan")

        tanggal_mulai = datetime.today().date() - pd.Timedelta(days=6)
        df_mingguan = df[df['tanggal_status'].dt.date >= tanggal_mulai]

        if not df_mingguan.empty:
            chart_data = (
                df_mingguan
                .groupby([df_mingguan['tanggal_status'].dt.strftime('%a'), 'label_sentimen'])
                .size()
                .reset_index(name='jumlah')
            )

            hari_urut = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            chart_data['tanggal_status'] = pd.Categorical(chart_data['tanggal_status'], categories=hari_urut, ordered=True)

            line_chart = alt.Chart(chart_data).mark_line(point=True).encode(
                x=alt.X('tanggal_status:N', title='Hari'),
                y=alt.Y('jumlah:Q', title='Jumlah Status'),
                color=alt.Color('label_sentimen:N', scale=alt.Scale(
                    domain=["POSITIF", "NETRAL", "NEGATIF"],
                    range=["#4CAF50", "#FFC107", "#F44336"]
                ), title="Sentimen"),
                tooltip=['tanggal_status:N', 'label_sentimen:N', 'jumlah:Q']
            ).properties(
                width=700,
                height=400,
                title="Tren Emosi Mingguan"
            )

            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.info("Belum ada status yang ditulis dalam 7 hari terakhir.")

    else:
        st.info("Belum ada riwayat tersedia.")
