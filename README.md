# Prediksi Radiasi Matahari ☀️

Aplikasi berbasis Streamlit untuk memprediksi radiasi matahari menggunakan model machine learning. Proyek ini bertujuan untuk membantu pengguna dalam menganalisis dan memprediksi potensi radiasi matahari untuk keperluan energi terbarukan atau penelitian.

## Fitur Utama 🚀
- **Prediksi Radiasi Matahari:** Masukkan data input untuk mendapatkan prediksi radiasi matahari secara real-time.
- **Analisis Akurasi Model:** Evaluasi performa model menggunakan metrik seperti R², Mean Absolute Error (MAE), dan Mean Squared Error (MSE).
- **Visualisasi Data:** Visualisasi data dan prediksi dengan grafik interaktif menggunakan Plotly.
- **User-Friendly Interface:** Aplikasi Streamlit yang responsif dan mudah digunakan.

## Teknologi yang Digunakan 🛠️
- **Python**: Bahasa pemrograman utama.
- **Streamlit**: Framework untuk membangun antarmuka aplikasi.
- **Scikit-learn**: Library untuk pembuatan dan evaluasi model machine learning.
- **Plotly & Seaborn**: Untuk visualisasi data interaktif dan informatif.

## Cara Menjalankan Aplikasi 🖥️
1. Clone repository ini ke perangkat Anda:
   ```bash
   git clone https://github.com/username/repository-name.git
2. pindah ke direktori proyek:
   cd repository-name
3. Install -r requirements.txt
   pip install -r requirements.txt
4. Jalankan aplikasi Streamlit :
   streamlit run pages/2_App.py

Struktur Proyek 📂
repository-name/
├── pages/
│   ├── 2_App.py            # Halaman utama aplikasi
│   ├── 3_About.py          # Halaman tentang proyek
│   ├── 4_Predict_From_CSV.py  # Halaman prediksi data CSV
├── save_folder/            # Folder untuk menyimpan model dan data
├── train_model.py          # Script untuk melatih model
├── scaler_DPA.pkl          # File scaler
├── model_solar.joblib      # Model machine learning terlatih
├── requirements.txt        # Daftar dependencies
└── README.md               # Dokumentasi proyek

Input dan Output 📝
Input: Data terkait kondisi lingkungan seperti temperatur, kelembapan, dan waktu (format CSV atau manual input).
Output: Prediksi radiasi matahari dalam satuan yang sesuai (misalnya W/m²).
URL Demo Aplikasi : https://solar-radiaton-prediction.streamlit.app/

Pengembangan Selanjutnya 🚧
Menambahkan prediksi untuk data real-time menggunakan API cuaca.
Memperluas dataset untuk meningkatkan akurasi model.
Mengintegrasikan hasil prediksi ke dalam dashboard energi terbarukan.

Kontributor 🤝
(luciferdana) - Developer.
Lisensi 📜
Proyek ini dilisensikan di bawah MIT License.

