import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os # Untuk memeriksa keberadaan file

# --- Konfigurasi Halaman Streamlit (Harus di awal) ---
st.set_page_config(
    page_title="Analisis Sentimen Kesehatan Mental",
    page_icon="🧠",
    layout="centered", # Konten akan berada di tengah halaman
    initial_sidebar_state="collapsed" # Sidebar tersembunyi secara default
)

# --- Fungsi untuk Memuat Model dan Alat NLP ---
@st.cache_resource # Gunakan cache_resource untuk memuat model hanya sekali
def load_nlp_resources():
    """Memuat model, TF-IDF vectorizer, dan LabelEncoder."""
    try:
        # Perhatikan: Disarankan untuk mengganti nama file "tfidf_vectorizer (1).pkl"
        # menjadi "tfidf_vectorizer.pkl" untuk konsistensi.
        model = joblib.load("logistic_regression_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer (1).pkl") # Pastikan nama file sudah diperbaiki
        label_encoder = joblib.load("label_encoder.pkl")
        return model, tfidf, label_encoder
    except FileNotFoundError as e:
        st.error(f"🚨 Error: File model atau vectorizer tidak ditemukan. Pastikan semua file (.pkl) berada di direktori yang sama dengan aplikasi Streamlit ini. Detail: {e}")
        st.stop() # Hentikan aplikasi jika file penting tidak ditemukan
    except Exception as e:
        st.error(f"🚨 Error saat memuat sumber daya NLP: {e}")
        st.stop()

model, tfidf, label_encoder = load_nlp_resources()

# --- Custom CSS untuk Tampilan Aplikasi (Tema Biru Modern) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Gaya untuk keseluruhan aplikasi */
    .stApp {
        background: #3b5998; /* Diubah menjadi biru Facebook */
        font-family: 'Inter', sans-serif; /* Font Inter untuk tampilan modern */
        min-height: 100vh; /* Tinggi minimal 100% viewport */
        color: #374151; /* Warna teks default */
    }
    
    /* Gaya untuk header utama aplikasi */
    .main-header {
        background: rgba(255, 255, 255, 0.95); /* Latar belakang semi-transparan */
        backdrop-filter: blur(20px); /* Efek blur pada latar belakang */
        border: 1px solid rgba(255, 255, 255, 0.2); /* Border tipis */
        border-radius: 20px; /* Sudut membulat */
        padding: 3rem 2rem; /* Padding internal */
        margin: 2rem auto; /* Margin atas/bawah dan tengah secara horizontal */
        text-align: center; /* Teks di tengah */
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1), /* Bayangan luar */
            0 0 0 1px rgba(255, 255, 255, 0.1) inset; /* Bayangan dalam */
        animation: fadeInUp 0.8s ease-out; /* Animasi muncul */
        position: relative;
        overflow: hidden; /* Pastikan shimmer tidak keluar */
    }
    
    /* Efek shimmer pada header */
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 3s infinite; /* Animasi shimmer */
    }
    
    /* Gaya untuk bagian logo di header */
    .logo-section {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-bottom: 2rem;
        color: #4a90e2;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Gaya untuk judul utama aplikasi */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2); /* Gradien teks */
        -webkit-background-clip: text; /* Untuk efek gradien pada teks */
        -webkit-text-fill-color: transparent; /* Untuk efek gradien pada teks */
        background-clip: text;
        margin: 1rem 0;
        letter-spacing: 3px;
        text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3); /* Bayangan teks */
        animation: pulse 2s infinite alternate; /* Animasi pulse */
    }
    
    /* Gaya untuk sub-judul */
    .subtitle {
        font-size: 1.3rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
        opacity: 0.8;
    }
    
    /* Gaya untuk card/kontainer utama input */
    .card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        transition: all 0.3s ease; /* Transisi halus saat hover */
        animation: slideUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .card:hover {
        transform: translateY(-5px); /* Efek naik saat hover */
        box-shadow: 
            0 35px 60px rgba(0, 0, 0, 0.15),
            0 0 0 1px rgba(255, 255, 255, 0.2) inset;
    }
    
    /* Garis gradien di atas card */
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 20px 20px 0 0;
    }
    
    /* Gaya untuk bagian hasil analisis */
    .result-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 800px;
        min-height: 400px; /* Tinggi minimal agar tidak terlalu kecil */
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.1),
            0 0 30px rgba(102, 126, 234, 0.1);
        animation: expandIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    /* Gaya untuk judul bagian */
    .section-title {
        font-size: 2rem;
        font-weight: 600;
        color: white; /* Diubah menjadi putih */
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
    }
    
    /* Garis bawah pada judul bagian */
    .section-title::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Gaya untuk area input teks (Streamlit textarea) */
    .stTextArea > div > div > textarea {
        min-height: 150px !important; /* Tinggi minimal textarea */
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        border: none !important; /* Hilangkan border default */
        background: transparent !important; /* Latar belakang transparan */
        resize: none !important; /* Nonaktifkan resize manual */
        outline: none !important; /* Hilangkan outline saat fokus */
        color: #374151; /* Warna teks input */
    }
    
    .stTextArea > div > div > textarea:focus {
        box_shadow: none !important; /* Hilangkan bayangan saat fokus */
    }

    /* Gaya untuk kontainer input teks */
    .stTextArea {
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        background: rgba(248, 250, 252, 0.8);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stTextArea:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.1);
    }
    
    /* Gaya untuk area teks yang ditampilkan (hasil analisis) */
    .text-display-area {
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        background: white; /* Diubah menjadi putih */
        backdrop-filter: blur(0px); /* Hapus blur jika ingin putih solid */
        font-size: 1.1rem;
        line-height: 1.6;
        color: #374151;
    }
    
    /* Gaya umum untuk tombol Streamlit */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25) !important;
        width: 100% !important; /* Lebar penuh */
        height: 50px !important; /* Tinggi tetap */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0.5rem 0 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35) !important;
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Gaya khusus untuk tombol kembali */
    .back-button-container .stButton > button {
        background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 1.5rem !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(107, 114, 128, 0.2) !important;
        margin-bottom: 1rem !important;
        width: auto !important; /* Lebar otomatis */
        min-width: 100px !important;
    }
    
    .back-button-container .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(107, 114, 128, 0.3) !important;
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%) !important;
    }
    
    /* Gaya untuk kotak hasil sentimen (positif) */
    .sentiment-positive {
        background: linear-gradient(135deg, #10b981, #34d399); /* Gradien hijau */
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        animation: bounceIn 0.6s ease-out; /* Animasi bounce */
    }
    
    /* Gaya untuk kotak hasil sentimen (negatif) */
    .sentiment-negative {
        background: linear-gradient(135deg, #ef4444, #f87171); /* Gradien merah */
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
        animation: bounceIn 0.6s ease-out;
    }
    
    /* Gaya untuk kotak hasil sentimen (netral) */
    .sentiment-neutral {
        background: linear-gradient(135deg, #f59e0b, #fbbf24); /* Gradien oranye */
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3);
        animation: bounceIn 0.6s ease-out;
    }
    
    /* Gaya untuk kotak rekomendasi/saran */
    .recommendation-box {
        background: white; /* Diubah menjadi putih */
        border-left: 5px solid #667eea; /* Border kiri berwarna */
        border-radius: 10px;
        padding: 1.5rem;
        margin: 2rem 0;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #374151;
        backdrop-filter: blur(0px); /* Hapus blur jika ingin putih solid */
        animation: slideInLeft 0.6s ease-out; /* Animasi slide dari kiri */
    }
    
    /* Gaya untuk footer */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 4rem;
        padding: 2rem;
        font-size: 1rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 1s ease-out;
    }
    
    /* --- Keyframe Animasi --- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes expandIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes bounceIn {
        0% { opacity: 0; transform: scale(0.3); }
        50% { opacity: 1; transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.02); }
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* --- Responsive Design (Media Queries) --- */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
            letter-spacing: 2px;
        }
        
        .card, .result-section {
            margin: 1rem;
            padding: 1.5rem;
        }
        
        .main-header {
            margin: 1rem;
            padding: 2rem 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Inisialisasi State Sesi Streamlit ---
# Digunakan untuk mengelola navigasi antar halaman dan menyimpan hasil analisis
if 'page' not in st.session_state:
    st.session_state.page = 'home' # Halaman default saat aplikasi dimulai
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None # Menyimpan hasil analisis teks

# --- Fungsi Halaman Utama (Home Page) ---
def show_home_page():
    """Menampilkan halaman selamat datang dengan tombol untuk memulai analisis."""
    st.markdown("""
        <div class="main-header">
            <div class="logo-section">
                <span>🧠 Analisis Sentimen</span>
            </div>
            <div class="main-title">ANALISIS SENTIMEN</div>
            <div class="subtitle">"SELAMAT DATANG DI SISTEM ANALISIS STATUS"</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Mengubah layout kolom untuk memindahkan tombol "Input Status" ke ujung kiri dan membuatnya lebih lebar
    col1, col2 = st.columns([3, 1]) # Kolom pertama 3x lebih lebar (untuk tombol), kolom kedua 1x lebih lebar (kosong)
    with col1: # Menempatkan tombol di kolom pertama (kiri)
        if st.button("📝 Input Status", key="input_status_btn", help="Klik untuk memulai analisis sentimen"):
            st.session_state.page = 'input' # Ubah state ke halaman input
            st.rerun() # Muat ulang aplikasi untuk menampilkan halaman baru
    st.markdown("<br>", unsafe_allow_html=True)

# --- Fungsi Halaman Input (Input Page) ---
def show_input_page():
    """Menampilkan halaman untuk memasukkan teks dan melakukan analisis."""
    st.markdown("""
        <div class="main-header">
            <div class="logo-section">
                <span>🧠 Analisis Sentimen</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Tombol kembali dengan styling khusus (ditempatkan di kolom terpisah agar CSS spesifik bisa diterapkan)
    col1, col2, col3 = st.columns([2, 6, 2])
    with col1:
        # Wrapper div untuk menerapkan CSS kustom ke tombol "Kembali"
        st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
        if st.button("← Kembali", key="back_btn", help="Kembali ke halaman utama"):
            st.session_state.page = 'home'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mengubah warna teks "MASUKKAN STATUS" menjadi putih
    st.markdown('<h2 class="section-title" style="color: white;">📝 MASUKKAN STATUS</h2>', unsafe_allow_html=True)
    
    # Area input teks
    text_input = st.text_area(
        "", # Label kosong karena styling sudah di CSS
        height=150, 
        placeholder="💭 Bagikan perasaan atau status Anda disini... Kami akan menganalisis sentimen dari kata-kata Anda.",
        label_visibility="collapsed", # Sembunyikan label default Streamlit
        key="text_input"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Mengubah layout kolom untuk memindahkan tombol "Analisis Sekarang" ke ujung kiri dan membuatnya lebih lebar
    col1, col2 = st.columns([3, 1]) # Kolom pertama 3x lebih lebar (untuk tombol), kolom kedua 1x lebih lebar (kosong)
    with col1: # Menempatkan tombol di kolom pertama (kiri)
        if st.button("🔍 Analisis Sekarang", key="analyze_btn", help="Mulai analisis sentimen"):
            if text_input.strip() == "": # Validasi input kosong
                st.error("⚠️ Mohon masukkan teks terlebih dahulu untuk dianalisis.")
            else:
                with st.spinner('🔄 Sedang menganalisis sentimen...'): # Tampilkan spinner saat analisis
                    # --- Proses Prediksi ---
                    # 1. Preprocessing (jika ada langkah selain yang ditangani TF-IDF)
                    #    Misalnya, text_input = text_input.lower() jika case folding tidak di TF-IDF
                    
                    # 2. Vectorization: Mengubah teks menjadi representasi numerik menggunakan TF-IDF
                    vectorized_text = tfidf.transform([text_input])
                    
                    # 3. Prediksi Model: Menggunakan model yang sudah dilatih
                    prediction = model.predict(vectorized_text)
                    
                    # 4. Inverse Transform: Mengubah label numerik kembali ke label string aslinya
                    #    Pastikan `label_encoder.classes_` sesuai dengan mapping sentimen Anda (e.g., [0: Negatif, 1: Netral, 2: Positif])
                    sentiment_label = label_encoder.inverse_transform(prediction)[0]
                    
                    # Simpan hasil ke session state
                    st.session_state.analysis_result = {
                        'text': text_input,
                        'sentiment': sentiment_label
                    }
                    st.session_state.page = 'result' # Ubah state ke halaman hasil
                    st.rerun() # Muat ulang aplikasi untuk menampilkan halaman hasil
    st.markdown("<br>", unsafe_allow_html=True)

# --- Fungsi Halaman Hasil (Result Page) ---
def show_result_page():
    """Menampilkan hasil analisis sentimen dan rekomendasi."""
    # Pastikan ada hasil analisis sebelum menampilkan halaman
    if st.session_state.analysis_result is None:
        st.session_state.page = 'home'
        st.rerun()
        return

    st.markdown("""
        <div class="main-header">
            <div class="logo-section">
                <span>🧠 Analisis Sentimen</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Mengubah warna teks "HASIL ANALISIS" menjadi putih
    st.markdown('<h2 class="section-title" style="color: white;">📊 HASIL ANALISIS</h2>', unsafe_allow_html=True)
    
    # Tampilkan teks yang dianalisis
    st.markdown(f"""
        <div class="text-display-area">
            <strong>📄 Status yang dianalisis:</strong><br><br>
            "{st.session_state.analysis_result['text']}"
        </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan hasil sentimen dengan styling dinamis berdasarkan label
    sentiment = st.session_state.analysis_result['sentiment'].lower() # Pastikan lowercase untuk perbandingan
    
    # --- Mapping Sentimen ke Kelas CSS dan Rekomendasi ---
    # PASTIKAN LABEL INI SESUAI DENGAN `label_encoder.classes_` ANDA
    if sentiment == "positif": # Sesuaikan dengan string label dari LabelEncoder Anda
        st.markdown(f"""
            <div class="sentiment-positive">
                🎉 Hasil Sentimen: POSITIF
            </div>
        """, unsafe_allow_html=True)
        
        recommendation = """
        <div class="recommendation-box">
            <strong>💡 Rekomendasi & Dukungan:</strong><br><br>
            ✨ Luar biasa! Energi positif Anda sangat menginspirasi<br>
            🌟 Terus jaga pikiran positif dan bagikan ke orang sekitar<br>
            🎯 Luangkan waktu untuk hal-hal yang membuat Anda bahagia<br>
            🚀 Momentum positif ini bisa dijadikan motivasi untuk pencapaian lebih besar
        </div>
        """
    elif sentiment == "negatif": # Sesuaikan dengan string label dari LabelEncoder Anda
        st.markdown(f"""
            <div class="sentiment-negative">
                💔 Hasil Sentimen: NEGATIF
            </div>
        """, unsafe_allow_html=True)
        
        recommendation = """
        <div class="recommendation-box">
            <strong>🤗 Saran & Dukungan:</strong><br><br>
            💪 Anda tidak sendirian, cobalah berbicara dengan orang terdekat<br>
            🩺 Pertimbangkan konsultasi dengan psikolog profesional<br>
            🧘 Luangkan waktu untuk self-care dan aktivitas yang menenangkan<br>
            ❤️ Ingat, Anda berharga dan pantas mendapatkan bantuan serta perhatian
        </div>
        """
    else: # Asumsi ini adalah "netral"
        st.markdown(f"""
            <div class="sentiment-neutral">
                ⚖️ Hasil Sentimen: NETRAL
            </div>
        """, unsafe_allow_html=True)
        
        recommendation = """
        <div class="recommendation-box">
            <strong>📌 Catatan & Saran:</strong><br><br>
            🔍 Status menunjukkan sentimen yang seimbang<br>
            👁️ Tetap pantau perasaan dan jangan ragu untuk berbagi cerita<br>
            ⚖️ Jaga keseimbangan hidup, tetaplah reflektif dan terbuka<br>
            🌱 Momen netral adalah kesempatan untuk introspeksi diri
        </div>
        """
    
    st.markdown(recommendation, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    # Mengubah layout kolom untuk memindahkan tombol "Kembali ke Awal" ke ujung kanan
    # col1 untuk "Analisis Ulang", col2 untuk space, col3 untuk "Kembali ke Awal"
    col1, col2, col3 = st.columns([1, 4, 1]) # Mengubah rasio kolom untuk membuat tombol lebih lebar dan "Kembali ke Awal" di ujung kanan
    with col1: # Tombol "Analisis Ulang" di kolom kiri
        if st.button("🔄 Analisis Ulang", key="analyze_again", help="Analisis status baru"):
            st.session_state.page = 'input'
            st.rerun()
    with col3: # Tombol "Kembali ke Awal" di kolom paling kanan
        if st.button("🏠 Kembali ke Awal", key="back_to_home", help="Kembali ke halaman utama"):
            st.session_state.page = 'home'
            st.session_state.analysis_result = None # Hapus hasil analisis sebelumnya
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

# --- Sembunyikan Elemen Streamlit Default ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;} /* Sembunyikan tombol deploy jika tidak diinginkan */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Logika Utama Aplikasi ---
def main():
    """Fungsi utama untuk mengelola alur aplikasi berdasarkan state sesi."""
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'input':
        show_input_page()
    elif st.session_state.page == 'result':
        show_result_page()

# --- Fungsi Footer ---
def show_footer():
    """Menampilkan footer aplikasi."""
    st.markdown("""
        <div class='footer'>
            © 2025 Sentimen Analisis Kesehatan Mental<br>
            <small>Dibuat dengan ❤️ menggunakan Machine Learning & NLP</small>
        </div>
    """, unsafe_allow_html=True)

# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    main()
    show_footer()
