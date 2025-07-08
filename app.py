import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import joblib
import random
import re
import io
import plotly.express as px
import time
import hashlib

st.set_page_config("Mental Health Sentiment App", page_icon="💬", layout="centered")

# -----------------------------
# Modern Font + Responsif Layout
# -----------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            font-size: 16px;
        }

        .main {
            padding-left: 1rem;
            padding-right: 1rem;
            transition: all 0.3s ease;
        }

        .sentiment-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f7f7f7;
            margin-top: 10px;
            margin-bottom: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }

        @media screen and (max-width: 600px) {
            .stButton > button {
                width: 100%;
            }
            .stTextInput > div, .stTextArea > div {
                width: 100% !important;
            }
        }

        @media (prefers-color-scheme: dark) {
            .sentiment-box {
                background-color: #1e1e1e;
                color: #f5f5f5;
            }
        }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# Load model, TF-IDF, encoder
# -----------------------------
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer (1).pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Fungsi bantu
# -----------------------------
def tampilkan_logo():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.image("sentimen.png", width=60)

def tampilkan_motivasi():
    motivasi_list = [
        "🌟 Kamu lebih kuat dari yang kamu pikirkan.",
        "💪 Setiap hari adalah kesempatan baru untuk memulai.",
        "✨ Jangan biarkan hari buruk merusak hidup yang indah.",
        "🧘 Ambil napas dalam, kamu akan baik-baik saja.",
        "🌈 Setelah hujan, selalu ada pelangi."
    ]
    motivasi = random.choice(motivasi_list)
    st.markdown(f"""<div class="sentiment-box"><strong>Motivasi Hari Ini:</strong><br>{motivasi}</div>""", unsafe_allow_html=True)

def tampilkan_motivasi_harian():
    today = datetime.today().date()
    if st.session_state.get("last_motivation_date") != today:
        st.session_state.last_motivation_date = today
        tampilkan_motivasi()

def check_autologout(timeout=900):
    now = time.time()
    last_active = st.session_state.get("last_active", now)
    if now - last_active > timeout:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.page = "login"
        st.rerun()
    else:
        st.session_state.last_active = now

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.page = "login"
    st.rerun()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        nama_lengkap TEXT,
        password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS status (
        id_status INTEGER PRIMARY KEY AUTOINCREMENT,
        isi_status TEXT,
        label_sentimen TEXT,
        kepercayaan REAL,
        tanggal_status DATE,
        id_user TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS journal (
        id_journal INTEGER PRIMARY KEY AUTOINCREMENT,
        isi TEXT,
        tanggal DATE,
        id_user TEXT)''')
    conn.commit()
    conn.close()

def simpan_user(username, nama_lengkap, password):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (username, nama_lengkap, password) VALUES (?, ?, ?)",
              (username, nama_lengkap, hash_password(password)))
    conn.commit()
    conn.close()

def validasi_login(username, password):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute("SELECT nama_lengkap, password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return "TIDAK_TERDAFTAR"
    elif row[1] != hash_password(password):
        return "PASSWORD_SALAH"
    else:
        return row[0]  # nama_lengkap


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

def simpan_journal(id_user, isi):
    conn = sqlite3.connect('sentimen.db')
    c = conn.cursor()
    c.execute("INSERT INTO journal (isi, tanggal, id_user) VALUES (?, ?, ?)",
              (isi, datetime.today().date(), id_user))
    conn.commit()
    conn.close()

def ambil_journal(id_user):
    conn = sqlite3.connect('sentimen.db')
    df = pd.read_sql_query("SELECT * FROM journal WHERE id_user = ? ORDER BY tanggal DESC", conn, params=(id_user,))
    conn.close()
    return df

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

# -----------------------------
# Inisialisasi session
# -----------------------------
init_db()
if "page" not in st.session_state:
    st.session_state.page = "login"

# -----------------------------
# Sidebar Navigasi
# -----------------------------
if st.session_state.page != "login":
    with st.sidebar:
        tampilkan_logo()
        st.title("🧭 Navigasi")
        st.button("🏠 Beranda", on_click=lambda: st.session_state.update(page="home"))
        st.button("✍️ Input Status", on_click=lambda: st.session_state.update(page="input"))
        st.button("📊 Riwayat", on_click=lambda: st.session_state.update(page="hasil"))
        st.button("📖 Journal", on_click=lambda: st.session_state.update(page="journal"))
        st.markdown("---")
        if "username" in st.session_state:
            if st.button("🚪 Logout"):
                logout()

# -----------------------------
# Halaman Login & Daftar
# -----------------------------
if st.session_state.page == "login":
    tampilkan_logo()
    st.title("🔐 Login Pengguna")

    tab1, tab2 = st.tabs(["🔓 Login", "📝 Daftar"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Masuk"):
            if not username or not password:
                st.warning("⚠️ Username dan password tidak boleh kosong.")
            else:
                hasil_login = validasi_login(username, password)
                
                if hasil_login == "TIDAK_TERDAFTAR":
                    st.warning("⚠️ Username belum terdaftar. Silakan daftar terlebih dahulu.")
                elif hasil_login == "PASSWORD_SALAH":
                    st.error("❌ Password salah. Coba lagi.")
                else:
                    st.session_state.username = username
                    st.session_state.nama = hasil_login
                    st.session_state.last_active = time.time()
                    st.session_state.last_motivation_date = None
                    st.session_state.page = "home"
                    st.success("✅ Login berhasil.")
                    st.rerun()


    with tab2:
        nama_lengkap = st.text_input("Nama Lengkap", key="reg_nama")
        reg_username = st.text_input("Username", key="reg_user")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Daftar"):
            if nama_lengkap and reg_username and reg_password:
                simpan_user(reg_username, nama_lengkap, reg_password)
                st.success("✅ Akun berhasil dibuat. Silakan login.")
            else:
                st.warning("Semua kolom wajib diisi.")

# -----------------------------
# Home Page
# -----------------------------
elif st.session_state.page == "home":
    if "username" not in st.session_state:
        st.session_state.page = "login"
        st.rerun()
    check_autologout()
    tampilkan_logo()
    tampilkan_motivasi_harian()
    st.title(f"Halo, {st.session_state.nama} 👋")
    st.write("Silakan pilih menu di sidebar.")

# -----------------------------
# Input Status Page
# -----------------------------
elif st.session_state.page == "input":
    if "username" not in st.session_state:
        st.session_state.page = "login"
        st.rerun()
    check_autologout()
    tampilkan_logo()
    st.title("📝 Tulis Status")
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

# -----------------------------
# Hasil dan Riwayat Page
# -----------------------------
elif st.session_state.page == "hasil":
    if "username" not in st.session_state:
        st.session_state.page = "login"
        st.rerun()
    check_autologout()
    tampilkan_logo()
    st.title("📊 Hasil Analisis")
    if "hasil_status" in st.session_state:
        st.subheader("Status Terakhir:")
        st.write(f'💬 **"{st.session_state.hasil_status}"**')
        st.write(f'📌 Sentimen: **{st.session_state.hasil_label}**')
        st.write(f'📈 Kepercayaan: **{st.session_state.hasil_conf}%**')
        rekomendasi = {
            "POSITIF": "😊 Pertahankan perasaan positifmu!",
            "NETRAL": "😐 Coba refleksi ringan dan jaga mood.",
            "NEGATIF": "😔 Menulis jurnal atau bicara ke teman bisa membantu."
        }
        st.info(rekomendasi.get(st.session_state.hasil_label, "❤️ Jaga kesehatan mentalmu ya!"))

    df = ambil_riwayat(st.session_state.username)
    if not df.empty:
        st.subheader("📈 Tren Harian")
        df['tanggal_status'] = pd.to_datetime(df['tanggal_status'])
        df_harian = df.groupby(['tanggal_status', 'label_sentimen']).size().reset_index(name='jumlah')
        fig = px.bar(
    df_harian,
    x='tanggal_status',
    y='jumlah',
    color='label_sentimen',
    color_discrete_map={
        'POSITIF': 'blue',
        'NETRAL': 'gray',
        'NEGATIF': 'red'
    },
    category_orders={"label_sentimen": ["POSITIF", "NETRAL", "NEGATIF"]},
    barmode='group'
)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧾 Riwayat")
        df['tanggal_status'] = df['tanggal_status'].dt.strftime('%d-%m-%Y')
        df_tampil = df.rename(columns={
            'tanggal_status': 'Tanggal',
            'isi_status': 'Status',
            'label_sentimen': 'Label Sentimen',
            'kepercayaan': 'Kepercayaan (%)'
        })[['Tanggal', 'Status', 'Label Sentimen', 'Kepercayaan (%)']]
        st.dataframe(df_tampil, use_container_width=True, hide_index=True)

        buffer = io.BytesIO()
        df_tampil.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button("📥 Unduh Riwayat ke Excel", data=buffer.getvalue(), file_name="riwayat_sentimen.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Belum ada riwayat tersedia.")

    if st.button("🗑 Hapus Semua Riwayat"):
        hapus_semua_riwayat(st.session_state.username)
        st.success("✅ Riwayat berhasil dihapus.")
        st.rerun()

# -----------------------------
# Journaling Page
# -----------------------------
elif st.session_state.page == "journal":
    if "username" not in st.session_state:
        st.session_state.page = "login"
        st.rerun()
    check_autologout()
    tampilkan_logo()
    tampilkan_motivasi_harian()
    st.title("📖 Journaling Harian")
    entry = st.text_area("Catatan hari ini:", height=200)
    if st.button("💾 Simpan Catatan"):
        if entry.strip():
            simpan_journal(st.session_state.username, entry)
            st.success("📝 Catatan berhasil disimpan!")
        else:
            st.warning("⚠️ Catatan tidak boleh kosong.")

    st.markdown("---")
    st.subheader("📚 Riwayat Catatan")
    df = ambil_journal(st.session_state.username)
    if not df.empty:
        df['tanggal'] = pd.to_datetime(df['tanggal']).dt.strftime('%d-%m-%Y')
        st.dataframe(df[['tanggal', 'isi']].rename(columns={'tanggal': 'Tanggal', 'isi': 'Catatan'}), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada catatan.")
