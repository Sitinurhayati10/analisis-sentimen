import streamlit as st import sqlite3 from datetime import datetime import pandas as pd import joblib import random import re import requests import urllib.parse

-----------------------------

1. Load Model dan Vectorizer

-----------------------------

model = joblib.load("logistic_regression_model.pkl") tfidf = joblib.load("tfidf_vectorizer (1).pkl") label_encoder = joblib.load("label_encoder.pkl")

-----------------------------

2. Fungsi Bantu

-----------------------------

def tampilkan_logo(): col1, col2, col3 = st.columns([1, 6, 1]) with col1: st.image("logo_sentimen.png", width=60)

def init_db(): conn = sqlite3.connect('sentimen.db') c = conn.cursor() c.execute(''' CREATE TABLE IF NOT EXISTS status ( id_status INTEGER PRIMARY KEY AUTOINCREMENT, isi_status TEXT, label_sentimen TEXT, kepercayaan REAL, tanggal_status DATE, id_user TEXT ) ''') conn.commit() conn.close()

def simpan_status(id_user, isi_status, label_sentimen, confidence): conn = sqlite3.connect('sentimen.db') c = conn.cursor() c.execute("INSERT INTO status (id_user, isi_status, label_sentimen, kepercayaan, tanggal_status) VALUES (?, ?, ?, ?, ?)", (id_user, isi_status, label_sentimen, confidence, datetime.today().date())) conn.commit() conn.close()

def ambil_riwayat(id_user): conn = sqlite3.connect('sentimen.db') df = pd.read_sql_query("SELECT * FROM status WHERE id_user = ? ORDER BY id_status DESC", conn, params=(id_user,)) conn.close() return df

def hapus_semua_riwayat(id_user): conn = sqlite3.connect('sentimen.db') c = conn.cursor() c.execute("DELETE FROM status WHERE id_user = ?", (id_user,)) conn.commit() conn.close()

def bersihkan_teks(teks): teks = teks.lower() teks = re.sub(r'[^a-zA-Z\s]', '', teks) return teks.strip()

def validasi_input(teks, minimal_kata=3): return len(teks.strip().split()) >= minimal_kata

def prediksi_sentimen(teks): if not validasi_input(teks): return "Terlalu pendek", 0

teks_bersih = bersihkan_teks(teks)
fitur = tfidf.transform([teks_bersih])
pred = model.predict(fitur)
prob = model.predict_proba(fitur).max() * 100
label = label_encoder.inverse_transform(pred)[0]
return label.upper(), round(prob, 2)

-----------------------------

3. Inisialisasi

-----------------------------

init_db() if "page" not in st.session_state: st.session_state.page = "login"

-----------------------------

4. Login Manual

-----------------------------

if st.session_state.page == "login": tampilkan_logo() st.title("\U0001F510 Login Pengguna")

username = st.text_input("Masukkan Nama Pengguna (manual):")
if username:
    st.session_state.username = username
    st.session_state.id_user = username
    st.session_state.page = "home"
    st.rerun()

st.markdown("---")
if st.button("\U0001F517 Login via Facebook"):
    st.session_state.page = "login_fb"
    st.rerun()

-----------------------------

5. Login Facebook

-----------------------------

elif st.session_state.page == "login_fb": tampilkan_logo() st.title("\U0001F517 Login dengan Facebook")

APP_ID = "2231988500574153"
APP_SECRET = "Analis Sentimen Status"  # Ganti dengan App Secret kamu
REDIRECT_URI = "https://sentimen-status-facebook.streamlit.app"

login_url = f"https://www.facebook.com/v19.0/dialog/oauth?client_id={APP_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}&scope=public_profile,user_posts"
st.markdown(f"[\U0001F510 Klik di sini untuk login dengan Facebook]({login_url})")

code = st.experimental_get_query_params().get('code', [None])[0]

if code:
    token_resp = requests.get("https://graph.facebook.com/v19.0/oauth/access_token", params={
        "client_id": APP_ID,
        "redirect_uri": REDIRECT_URI,
        "client_secret": APP_SECRET,
        "code": code
    })

    if token_resp.status_code == 200:
        access_token = token_resp.json().get("access_token")
        # Ambil info pengguna
        user_info = requests.get("https://graph.facebook.com/me", params={
            "access_token": access_token,
            "fields": "id,name"
        }).json()

        st.session_state.username = user_info.get("name")
        st.session_state.id_user = user_info.get("id")
        st.session_state.page = "home"
        st.experimental_rerun()
    else:
        st.error("Gagal mendapatkan token akses Facebook. Silakan coba lagi.")
