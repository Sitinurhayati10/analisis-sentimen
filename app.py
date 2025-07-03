import streamlit as st
import sqlite3
import requests
import joblib
import pandas as pd
import re
from datetime import datetime
import urllib.parse

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

    if "fb_token" not in st.session_state:
        st.markdown("---")
        st.subheader("Atau login via Facebook:")
        APP_ID = st.secrets["APP_ID"]
        REDIRECT_URI = st.secrets["REDIRECT_URI"]
        login_url = f"https://www.facebook.com/v19.0/dialog/oauth?client_id={APP_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}&scope=public_profile"
        st.markdown(f"[🔗 Klik untuk login dengan Facebook]({login_url})")

    code = st.query_params.get("code", None)
    if isinstance(code, list):
        code = code[0]
    
    if code:
        APP_ID = st.secrets["APP_ID"]
        APP_SECRET = st.secrets["APP_SECRET"]
        REDIRECT_URI = st.secrets["REDIRECT_URI"]

        token_resp = requests.get("https://graph.facebook.com/v19.0/oauth/access_token", params={
            "client_id": APP_ID,
            "redirect_uri": REDIRECT_URI,
            "client_secret": APP_SECRET,
            "code": code
        }).json()

        st.write("DEBUG TOKEN:", token_resp)  # Tambahkan ini

        access_token = token_resp.get("access_token")
        if access_token:
            profile = requests.get(f"https://graph.facebook.com/me?fields=id,name&access_token={access_token}").json()
            user_id = profile.get("id")
            user_name = profile.get("name")

            st.session_state.username = user_id
            st.session_state.fb_token = access_token
            st.success(f"✅ Login berhasil sebagai {user_name}")
            st.session_state.page = "fb_status"
            st.rerun()
        else:
            st.error("❌ Gagal login ke Facebook. Periksa konfigurasi App ID dan Secret.")

# ========== 5. Facebook Status Page ==========
elif st.session_state.page == "fb_status":
    tampilkan_logo()
    st.title("📥 Status Facebook Anda")

    token = st.session_state.fb_token
    user_id = st.session_state.username

    try:
        posts = requests.get(f"https://graph.facebook.com/me/feed?fields=message,created_time&access_token={token}").json()
        for post in posts.get("data", []):
            teks = post.get("message", "")
            if teks:
                st.write(f"💬 {teks}")
                label, conf = prediksi_sentimen(teks)
                st.write(f"📌 Sentimen: **{label}**, Kepercayaan: {conf}%")
                simpan_status(user_id, teks, label, conf)

        st.success("✅ Semua status publik berhasil dianalisis dan disimpan.")

    except Exception as e:
        st.error("⚠️ Gagal mengambil status dari Facebook.")
        st.error(str(e))

    if st.button("🔁 Lihat Riwayat"):
        st.session_state.page = "hasil"
        st.rerun()

# ========== 6. Home Page ==========
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

# ========== 7. Input Status Page ==========
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

# ========== 8. Hasil dan Riwayat ==========
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

    st.subheader("🕒 Riwayat Status Anda")
    if st.button("🗑 Hapus Semua Riwayat"):
        hapus_semua_riwayat(st.session_state.username)
        st.success("✅ Riwayat berhasil dihapus.")
        st.rerun()

    df = ambil_riwayat(st.session_state.username)
    if not df.empty:
        df['tanggal_status'] = pd.to_datetime(df['tanggal_status']).dt.strftime('%d-%m-%Y')
        df_tampil = df.rename(columns={
            'tanggal_status': 'Tanggal',
            'isi_status': 'Status',
            'label_sentimen': 'Label Sentimen',
            'kepercayaan': 'Kepercayaan (%)'
        })[['Tanggal', 'Status', 'Label Sentimen', 'Kepercayaan (%)']]
        st.dataframe(df_tampil, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada riwayat tersedia.")
