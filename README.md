# Credit Risk Prediction: Lending Club Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi risiko gagal bayar pinjaman. Menggunakan data historis dari Lending Club, kami mengembangkan model klasifikasi untuk membedakan antara "Good Loan" dan "Bad Loan" (Default/Charged-off). Solusi ini membantu institusi keuangan mengotomatisasi penilaian kredit dan meminimalkan kerugian finansial.

## 📉 Problem Statement
Evaluasi kredit manual memakan waktu dan rentan terhadap kesalahan manusia. Dengan tingkat gagal bayar sekitar **10,93%** dalam dataset ini, diperlukan sistem otomatis yang dapat mengidentifikasi peminjam berisiko tinggi sebelum pinjaman diberikan.

## 🛠️ Tech Stack
- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Algorithm:** Random Forest Classifier

## 🚀 Workflow
1. **Data Cleaning:** Menangani nilai yang hilang dan mereduksi 75 fitur awal menjadi 46 variabel esensial.
2. **Feature Engineering:** - Mengonversi data kategorikal (Grade, Term, Emp Length) menjadi format numerik.
   - Membuat fitur temporal seperti `mths_since_issue_d` untuk menangkap tren berbasis waktu.
3. **Exploratory Data Analysis (EDA):** Mengidentifikasi ketidakseimbangan data dan korelasi utama.
4. **Modeling:** Melatih model Random Forest menggunakan `class_weight='balanced'` untuk menangani ketidakseimbangan kelas target.
5. **Evaluation:** Menilai model menggunakan Accuracy, Recall, dan Confusion Matrix.

## 📊 Results
Model mencapai performa luar biasa dalam memprediksi gagal bayar pinjaman:
- **Overall Accuracy:** 98.23%
- **Recall (Deteksi Bad Loan):** 86%

## 💾 Dataset Information
Karena ukuran file dataset asli (`loan_data_2007_2014.csv`) yang terlalu besar untuk diunggah langsung ke GitHub, Anda dapat mengakses dan mengunduh dataset tersebut melalui tautan resmi berikut:

🔗 **Link Dataset:** [https://www.rakamin.com/virtual-internship-experience/task/id-x-partners-data-scientist-pbi/36240](https://www.rakamin.com/virtual-internship-experience/task/id-x-partners-data-scientist-pbi/36240)

## 📂 Repository Structure
- `Analisis_Credit_Risk.ipynb`: Analisis lengkap, visualisasi, dan proses langkah demi langkah.
- `Analisis_Credit_Risk.py`: Script Python yang dapat dieksekusi untuk pipeline model.
- `LCDataDictionary.xlsx.pdf`: Referensi untuk definisi fitur.
- `README.md`: Dokumentasi proyek.

## 💡 Business Recommendation
- **Otomatisasi:** Gunakan model untuk menyetujui aplikasi berisiko rendah secara otomatis.
- **Peringatan Dini:** Tandai peminjam berisiko tinggi untuk tinjauan manual atau persyaratan yang lebih ketat.
- **Mitigasi Risiko:** Implementasi model ini berpotensi mengurangi kerugian kredit hingga 86% melalui deteksi "Bad Loan" yang lebih baik.

---
*Dikembangkan sebagai bagian dari Virtual Internship Experience Data Scientist di IDX Partners.*
