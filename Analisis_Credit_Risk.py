#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data - menggunakan low_memory=False karena dataset memiliki banyak kolom
df = pd.read_csv('loan_data_2007_2014.csv', low_memory=False)

# Cek 5 baris pertama
df.head()


# In[2]:


# 1. Hapus kolom yang memiliki missing values lebih dari 50%
# Dataset ini punya banyak kolom yang benar-benar kosong di tahun tersebut
df_clean = df.dropna(thresh=len(df) * 0.5, axis=1)

# 2. Hapus kolom identitas atau teks yang tidak bisa diolah model
# 'Unnamed: 0' adalah indeks lama, 'id' & 'member_id' hanya nomor urut
cols_to_drop = ['Unnamed: 0', 'id', 'member_id', 'url', 'title', 'zip_code', 'policy_code', 'application_type']
df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

print(f"Jumlah kolom awal: {df.shape[1]}")
print(f"Jumlah kolom setelah dibersihkan: {df_clean.shape[1]}")


# In[3]:


# 1. Melihat distribusi unik dari loan_status
print("Distribusi Status Pinjaman Asli:")
print(df_clean['loan_status'].value_counts())

# 2. Menentukan daftar status yang masuk kategori 'Bad Loan' (Risiko Tinggi)
# Status ini menunjukkan peminjam yang gagal memenuhi kewajiban tepat waktu
bad_loan_status = [
    'Charged Off', 
    'Default', 
    'Does not meet the credit policy. Status:Charged Off', 
    'Late (31-120 days)'
]

# 3. Membuat kolom baru 'target'
# Nilai 1 untuk Bad Loan, Nilai 0 untuk Good Loan
df_clean['target'] = np.where(df_clean['loan_status'].isin(bad_loan_status), 1, 0)

# 4. Visualisasi untuk bahan presentasi/infografis
plt.figure(figsize=(7,5))
sns.countplot(x='target', data=df_clean, hue='target', palette='Set1', legend=False)
plt.title('Perbandingan Good Loan (0) vs Bad Loan (1)')
plt.xlabel('Target (0 = Good, 1 = Bad)')
plt.ylabel('Jumlah Peminjam')
plt.show()

# 5. Cek persentase perbandingannya
print(f"Proporsi Target:\n{df_clean['target'].value_counts(normalize=True) * 100}")


# In[4]:


# 1. Mengubah 'term' (jangka waktu) menjadi angka (misal: ' 36 months' -> 36)
df_clean['term'] = df_clean['term'].str.extract(r'(\d+)').astype(int)

# 2. Mengubah 'emp_length' (lama bekerja) menjadi angka
# Kita asumsikan '< 1 year' sebagai 0 dan '10+ years' sebagai 10
df_clean['emp_length'] = df_clean['emp_length'].str.extract(r'(\d+)').fillna(0).astype(int)

# 3. Mengubah 'grade' menjadi angka agar memiliki urutan risiko (A paling rendah risiko, G paling tinggi)
grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
df_clean['grade'] = df_clean['grade'].map(grade_map)

print("Berhasil mengolah kolom term, emp_length, dan grade.")
df_clean[['term', 'emp_length', 'grade']].head()


# In[5]:


# 1. Mengubah kolom 'issue_d' menjadi format datetime
df_clean['issue_d'] = pd.to_datetime(df_clean['issue_d'], format='%b-%y')

# 2. Menghitung durasi (dalam bulan) dari tanggal pinjaman sampai Desember 2025
# Semakin lama usia pinjaman, semakin banyak data histori pembayarannya
df_clean['mths_since_issue_d'] = ((pd.to_datetime('2025-12-01') - pd.to_datetime(df_clean['issue_d'])).dt.days / 30.44).round()

# 3. Hapus kolom tanggal asli karena sekarang sudah digantikan oleh versi angka
# Kita juga hapus kolom loan_status karena sudah kita ubah jadi kolom 'target'
cols_to_drop_date = ['issue_d', 'loan_status']
df_clean = df_clean.drop(columns=[c for c in cols_to_drop_date if c in df_clean.columns])

print("Kolom tanggal berhasil diubah menjadi durasi bulan.")
df_clean[['mths_since_issue_d']].head()


# In[6]:


# Hapus kolom tanggal asli dan loan_status
cols_to_drop_date = ['issue_d', 'loan_status']
df_clean = df_clean.drop(columns=[c for c in cols_to_drop_date if c in df_clean.columns])

print("Perbaikan Berhasil! Kolom tanggal kini sudah menjadi durasi bulan.")
print(df_clean[['mths_since_issue_d']].head())


# In[7]:


# 1. Menghitung korelasi numerik dengan target
correlations = df_clean.select_dtypes(exclude=['object']).corr()['target'].sort_values(ascending=False)

# 2. Ambil 10 faktor teratas (selain target itu sendiri yang nilainya pasti 1.0)
top_indicators = correlations.iloc[1:11]

# 3. Visualisasi untuk bahan Infografis
plt.figure(figsize=(10, 6))
sns.barplot(x=top_indicators.values, y=top_indicators.index, palette='viridis')
plt.title('Top 10 Indikator Risiko Kredit (Korelasi terhadap Bad Loan)')
plt.xlabel('Nilai Korelasi')
plt.show()

print("Indikator Risiko Teratas (Korelasi terhadap Bad Loan):")
print(top_indicators)


# In[8]:


from sklearn.model_selection import train_test_split

# 1. Menentukan Fitur (X) dan Target (y)
# X adalah semua kolom angka kecuali target, y adalah kolom target (0/1)
X = df_clean.select_dtypes(exclude=['object']).drop(columns=['target'])
y = df_clean['target']

# 2. Mengisi sisa nilai kosong (NaN) dengan median
# Ini langkah keamanan terakhir agar model tidak error saat membaca data
X = X.fillna(X.median())

# 3. Membagi data menjadi Training dan Testing
# stratify=y sangat penting agar proporsi 10.9% Bad Loan tetap sama di kedua bagian
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("--- Data Berhasil Dibagi ---")
print(f"Jumlah baris Data Training (untuk belajar): {X_train.shape[0]}")
print(f"Jumlah baris Data Testing (untuk ujian): {X_test.shape[0]}")
print(f"Jumlah kolom fitur: {X_train.shape[1]}")


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Inisialisasi Model
# n_estimators=100: Menggunakan 100 pohon keputusan
# class_weight='balanced': Memberitahu model untuk lebih teliti pada nasabah 'Bad Loan' karena jumlahnya sedikit
rf_model = RandomForestClassifier(n_estimators=100, 
                                  max_depth=10, 
                                  random_state=42, 
                                  class_weight='balanced')

# 2. Melatih Model (Proses Belajar)
print("Sedang melatih model... (mungkin butuh waktu 1-2 menit)")
rf_model.fit(X_train, y_train)

# 3. Melakukan Prediksi (Ujian)
y_pred = rf_model.predict(X_test)

# 4. Evaluasi Hasil
print("\n--- HASIL EVALUASI MODEL ---")
print(f"Akurasi Model: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))


# In[10]:


# Melihat variabel yang paling berpengaruh menurut model Random Forest
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='barh', color='darkblue')
plt.title('Top 10 Fitur Paling Berpengaruh dalam Pengambilan Keputusan Model')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:




