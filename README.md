# Laporan Proyek Machine Learning - M Adam Raya

## Domain Proyek
Perkembangan teknologi dan ketersediaan data dalam jumlah besar telah membuka peluang besar dalam penerapan machine learning di berbagai bidang, termasuk sektor keuangan. Salah satu tantangan penting di sektor ini adalah kemampuan untuk memprediksi pergerakan harga atau tren pasar berdasarkan data historis. Prediksi yang akurat tidak hanya membantu investor dalam mengambil keputusan strategis, tetapi juga mendukung perusahaan dalam merancang kebijakan manajemen risiko yang lebih baik.

Machine learning menjadi solusi yang sangat menjanjikan karena kemampuannya dalam mengenali pola dari data besar yang kompleks. Dengan algoritma yang tepat, model machine learning dapat mempelajari hubungan antar variabel, bahkan yang tidak terlihat oleh analis tradisional.

Proyek ini bertujuan untuk membandingkan performa dua algoritma machine learning, yaitu Random Forest dan Logistic Regression, dalam memprediksi suatu target variabel berdasarkan fitur numerik dan waktu dari dataset. Melalui proses pelatihan model, evaluasi performa menggunakan metrik seperti akurasi, precision, recall, dan f1-score, serta visualisasi seperti confusion matrix dan barplot, proyek ini akan memberikan wawasan tentang algoritma mana yang lebih optimal dalam konteks data yang digunakan.

Dengan mengevaluasi dan membandingkan hasil dari kedua model tersebut, diharapkan proyek ini dapat memberikan gambaran yang lebih jelas mengenai efektivitas algoritma dalam prediksi berbasis data historis, serta memberikan dasar bagi pengembangan model prediksi yang lebih akurat dan andal di masa mendatang.

## Business Understanding

### Problem Statements
Dalam sebuah proyek berbasis data, langkah awal yang sangat penting adalah merumuskan pernyataan masalah secara jelas dan terukur. Berdasarkan latar belakang yang telah disampaikan, berikut adalah beberapa pernyataan masalah yang menjadi dasar dalam proyek ini:
1. Bagaimana membangun model prediksi yang mampu memetakan hubungan antara fitur historis harga saham (seperti Open, High, Low, Close, Volume) terhadap target variabel secara akurat?
2. Model machine learning mana yang memberikan performa terbaik dalam melakukan prediksi pada dataset ini, antara Random Forest dan Logistic Regression?

### Goals

Tujuan dari proyek ini adalah :
1. Melakukan eksplorasi dan analisis data untuk memahami struktur, pola, dan distribusi fitur numerik historis, lalu melakukan preprocessing seperti normalisasi (MinMaxScaler), encoding variabel waktu (tahun/bulan), serta pemisahan fitur penting untuk model prediksi, dan Membangun model machine learning dengan algoritma yang mampu menangkap hubungan antar variabel meskipun bersifat non-linear.
2. Menerapkan dua algoritma berbeda: Random Forest (model berbasis ensemble decision tree) dan Logistic Regression (model linear klasifikasi) dan Menampilkan perbandingan hasil evaluasi secara visual (grafik barplot).

### Solution statements
1. Melakukan proses Exploratory Data Analysis (EDA) untuk memahami karakteristik data historis seperti harga pembukaan, penutupan, volume perdagangan, serta pola pergerakan nilai saham. Analisis korelasi dilakukan untuk mengidentifikasi fitur-fitur yang redundan, serta mengukur seberapa besar hubungan antar fitur terhadap target variabel yang akan diprediksi.
2. Menggunakan 2 model machine learning yaitu Random Forest dan Logistic Regression untuk membandingkan efektivitas dalam melakukan prediksi target berdasarkan data historis. Logistic Regression digunakan sebagai baseline model, sedangkan Random Forest digunakan sebagai model non-linear yang lebih kompleks untuk menguji peningkatan akurasi.
3. Melakukan hyperparameter tuning pada kedua algoritma untuk menemukan kombinasi parameter terbaik seperti n_estimators, max_depth, dan criterion pada Random Forest serta penalty, solver, dan C pada Logistic Regression guna meningkatkan performa model secara signifikan.
4. Menggunakan confusion matrix, accuracy, precision, recall, dan f1-score sebagai metrik evaluasi pada masing-masing model machine learning untuk menilai performa klasifikasi, dan menentukan model terbaik berdasarkan skor metrik tertinggi.
5. Menampilkan perbandingan performa model dalam bentuk visualisasi barplot, untuk memberikan gambaran yang lebih jelas terhadap akurasi masing-masing model, sehingga proses pengambilan keputusan terhadap model terbaik menjadi lebih terukur dan objektif.

## Data Understanding
Dataset yang digunakan untuk mempredisksi kinerja siswa diambil dari platform kaggle yang dapat diakses pada tautan berikut (https://www.kaggle.com/datasets/caesarmario/samsung-electronics-stock-historical-price) yang dipublikasikan oleh Mario Caesar. Kumpulan data ini berisi informasi lengkap history harga saham dari tahun 2019 sampai tahun 2025.
Infromasi dataset tersebut dapat dilihat pada gambar dibawah ini:

![image](https://github.com/user-attachments/assets/13fafd35-6ca1-4469-88ac-48799a638c95)

## Variabel pada dataset adalah sebagai berikut:
| Kolom       | Deskripsi                                                                         |
| ----------- | --------------------------------------------------------------------------------- |
| `Date`      | Tanggal pencatatan data saham                                                     |
| `Open`      | Harga pembukaan saham pada hari tersebut                                          |
| `High`      | Harga tertinggi yang dicapai pada hari tersebut                                   |
| `Low`       | Harga terendah yang dicapai pada hari tersebut                                    |
| `Close`     | Harga penutupan saham pada hari tersebut                                          |
| `Adj Close` | Harga penutupan yang telah disesuaikan (misalnya, untuk stock split atau dividen) |
| `Volume`    | Jumlah saham yang diperdagangkan pada hari tersebut                               |

## Data loading

![image](https://github.com/user-attachments/assets/72f313c2-25b5-4f54-a226-a86f229c6668)

Dari tabel diatas terdapat 1505 baris dan 7 kolom pada dataset.

## Exploratory Data Analysis

![image](https://github.com/user-attachments/assets/79eb950d-8b97-4a74-a5eb-6bc7abcdee55)

Dari hasil informasi terlihat bahwa :
- Terdapat 1 kolom dengan tipe object, yaitu : Date.
  - Kolom ini harusnya bertipe datetime karena mempresentasikan tanggal, bukan tipe data object yang digunakan untuk teks (string)
- Terdapat 5 kolom dengan tipe float64, yaitu : Open, High, Low, Close, dan Adj Close.
  - Tipe data sudah sesuai dengan isi dari kolomya.
- Terdapat 1 kolom dengan tipe int64, yaitu : volume.
  - Tipe data sudah sesuai dengan isi dari kolomya.

## Exploratory Data Analysis - Univariate Analysis
### Fitur Numerik

![image](https://github.com/user-attachments/assets/e1d60487-753c-4b47-badc-0b9d848a7b75)

Berdasarkan grafik distribusi data dari numerical_feature yang ditampilkan, dapat dilihat bahwa :
1. Rentang nilai dari setiap variabel (Open, High, Low, Close) memiliki rentang nilai yang serupa, yaitu 40.000 hingga 90.000. Namun pada Adj_Close berkisar 30.000 hingga 80.000, sedangkan Volume berkisar 0 sampai 35.000.000.
2. Pada Adj_Close menunjukkan bahwa harga penutupan yang disesuaikan lebih stabil atau tidak mencapai nilai ekstrem seperti Close (90.000).
3. Pada Volume tidak ada rentang nilai yang jelas, tetapi jika dianggap dalam skala yang berbeda (misalnya jutaan), distribusi mungkin sangat miring (right-skewed) karena volume perdagangan seringkali memiliki outlier tinggi.

### Fitur Datetime

![image](https://github.com/user-attachments/assets/1afe3b86-b7ab-4adb-a0e8-f9705d078a45)

Berdasarkan grafik dari jumlah data per tahun , dapat disimpulkan bahwa :
- 2019 terdapat 246 data
- 2020 terdapat 209 data
- 2021 terdapat 222 data
- 2022 terdapat 242 data
- 2023 terdapat 245 data
- 2024 terdapat 208 data
- 2025 terdapat 28 data

## Exploratory Data Analysis - Multivariate Analysis
### Melihat korelasi variabel numerik dengan menggunakan Heatmap

![image](https://github.com/user-attachments/assets/ce9ba542-d6b7-45f6-acf5-726756bb0522)

Berdasarkan gambar diatas, terlihat bahwa :
1. Fitur Open, High, Low, Close, Adj Close punya korelasi mendekati 1, artinya mereka mengandung informasi yang hampir sama.
2. Volume perdagangan tidak berkorelasi langsung dengan perubahan harga saham.

#### visualisasi time series

![image](https://github.com/user-attachments/assets/fd5cbe54-f125-42df-87c1-17af3bc8f62e)

Dari tahun 2018 hingga awal 2021, terlihat tren kenaikan harga saham yang signifikan, terutama lonjakan tajam di akhir 2020 hingga awal 2021. Setelah mencapai puncak di awal 2021 (~90.000), harga mulai mengalami fluktuasi dan penurunan bertahap.

## Data Preparation
Pada tahap ini kita akan melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahap persiapan data perlu dilakukan, yaitu :

1. Drop kolom yang tidak diperlukan.
2. Pembagian dataset dengan fungsi train_test_split dari library sklearn.

## Modeling
### Random Forest
Algoritma pembelajaran ensemble yang sangat populer untuk tugas klasifikasi dan regresi. Ini bekerja dengan membuat sejumlah pohon keputusan selama pelatihan dan menggabungkan hasilnya (melalui voting untuk klasifikasi atau rata-rata untuk regresi) untuk meningkatkan akurasi dan mengurangi overfitting.
Pada pemodelan ini, Random Forest diimplementasikan menggunakan RandomForestClassifier dari library sklearn.ensemble dengan memasukkan X_train dan y_train untuk melatih model, lalu menggunakan X_test dan y_test untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah n_estimators yaitu jumlah tree yang akan dibuat, criterion yaitu fungsi untuk menentukan kualitas splitting data, max_depth yaitu kedalaman maksimum setiap tree, dan random_state yaitu mengontrol seed acak yang diberikan pada setiap iterasi. Pada proyek ini, parameter yang digunakan adalah n_estimators=50, max_depth=16, random_state=42.

### Logistic Regression
Logistic Regression adalah salah satu algoritma machine learning yang paling dasar namun sangat kuat untuk tugas klasifikasi. Meskipun namanya mengandung kata "regression", algoritma ini digunakan untuk memprediksi kelas (bukan nilai kontinu) dengan menggunakan fungsi logistik (sigmoid) untuk mengubah output prediksi menjadi nilai probabilitas antara 0 dan 1.
Model ini bekerja dengan mengasumsikan hubungan linear antara fitur input dan log odds dari kelas target. Logistic Regression sangat efektif untuk masalah klasifikasi biner maupun multi-kelas dengan data yang relatif bersih dan linier secara separabilitas.
Pada pemodelan ini, Logistic Regression diimplementasikan menggunakan LogisticRegression dari library sklearn.linear_model. Model ini dilatih menggunakan data latih (X_train dan y_train) dan diuji dengan data uji (X_test dan y_test) yang belum pernah dilihat model sebelumnya.
Parameter yang digunakan pada model ini adalah penalty='l2', C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42.


## Evaluation
### Menghitung metrik precision, recall, F1-score, dan support.

![image](https://github.com/user-attachments/assets/c36208dc-ba27-4392-9a60-ce3b3c2567e1)

1. Precision
Presisi mengukur seberapa akurat prediksi positif model. Artinya, dari semua prediksi yang diklaim positif, berapa banyak yang benar-benar positif.

![image](https://github.com/user-attachments/assets/2fee8666-45a1-4ffe-aa28-22a6ffa0bca3)

2. Recall
Recall mengukur seberapa baik model menemukan semua kasus positif yang sebenarnya. Dari semua data yang benar-benar positif, berapa banyak yang berhasil ditemukan model.

![image](https://github.com/user-attachments/assets/5b44587a-fefa-4250-ae32-abb8a724f522)

3. F1-Score
F1-score adalah harmonic mean dari precision dan recall. Ini berguna jika kita ingin keseimbangan antara precision dan recall, terutama saat data tidak seimbang.

![image](https://github.com/user-attachments/assets/c0c226e0-4da1-4e1c-951d-9e3a2d9c5c87)

4. Support
Support adalah jumlah data aktual (ground truth) untuk setiap kelas. Ini hanya menunjukkan berapa banyak sampel dari setiap kelas dalam data.

5. Accuracy
Akurasi adalah metrik evaluasi yang mengukur proporsi prediksi yang benar dibandingkan dengan total keseluruhan prediksi.

![image](https://github.com/user-attachments/assets/67b7f97e-cb0a-4dd4-8b53-60f5cc21c8dd)

### Melihat perbandingan akurasi model dengan grafik barplot

![image](https://github.com/user-attachments/assets/4fd6f46f-d8db-4aaf-be0b-82254f5fecaf)

Dari grafik barplot yang ditampilkan, terlihat bahwa model menggunakan algoritma Random Forest lebih tinggi akurasinya yakni 0.9571 dibandingkan dengan model yakni Naive Bayes = 0.7964 untuk memprediksi harga saham setiap tahun.

## Referensi
1. Permana, N. A., & Bunyamin, H. (2024). Perbandingan Logistic Regression dengan Random Forest dalam Memprediksi Sentimen Pada IMDb Moview Review. Jurnal STRATEGI-Jurnal Maranatha, 6(2), 391-399.
2. Febriyanti, N. R., Kusrini, K., & Hartanto, A. D. (2025). Analisis Perbandingan Algoritma SVM, Random Forest dan Logistic Regression untuk Prediksi Stunting Balita. Edumatic: Jurnal Pendidikan Informatika, 9(1), 149-158.

