# Laporan Proyek Machine Learning - M Adam Raya

## Domain Proyek
Perkembangan teknologi dan ketersediaan data dalam jumlah besar telah membuka peluang besar dalam penerapan machine learning di berbagai bidang, termasuk sektor keuangan. Salah satu tantangan penting di sektor ini adalah kemampuan untuk memprediksi pergerakan harga atau tren pasar berdasarkan data historis. Prediksi yang akurat tidak hanya membantu investor dalam mengambil keputusan strategis, tetapi juga mendukung perusahaan dalam merancang kebijakan manajemen risiko yang lebih baik.

Machine learning menjadi solusi yang sangat menjanjikan karena kemampuannya dalam mengenali pola dari data besar yang kompleks. Dengan algoritma yang tepat, model machine learning dapat mempelajari hubungan antar variabel, bahkan yang tidak terlihat oleh analis tradisional.

Proyek ini bertujuan untuk membandingkan performa dua algoritma machine learning, yaitu Random Forest dan Logistic Regression, dalam memprediksi suatu target variabel berdasarkan fitur numerik dan waktu dari dataset. Melalui proses pelatihan model, evaluasi performa menggunakan metrik seperti akurasi, precision, recall, dan f1-score, serta visualisasi seperti confusion matrix dan barplot, proyek ini akan memberikan wawasan tentang algoritma mana yang lebih optimal dalam konteks data yang digunakan.

Dengan mengevaluasi dan membandingkan hasil dari kedua model tersebut, diharapkan proyek ini dapat memberikan gambaran yang lebih jelas mengenai efektivitas algoritma dalam prediksi berbasis data historis, serta memberikan dasar bagi pengembangan model prediksi yang lebih akurat dan andal di masa mendatang.

## Business Understanding

### Problem Statements
Berdasarkan kode yang menganalisis data harga saham dan mencoba memprediksi "Year" (meskipun ini mungkin indikator untuk periode waktu, bukan prediksi tahun spesifik secara harfiah, tetapi kita akan fokus pada prediksi pergerakan terkait waktu), berikut adalah pernyataan masalah yang relevan:

- Volatilitas harga saham: Harga saham cenderung berfluktuasi secara signifikan dari waktu ke waktu, membuat sulit bagi investor untuk memprediksi pergerakan harga di masa depan.
- Kurangnya wawasan dari data historis: Data historis harga saham mengandung informasi berharga tentang pola dan tren, namun tanpa analisis yang tepat, wawasan ini sulit untuk diekstraksi dan dimanfaatkan.
- Kebutuhan alat bantu pengambilan keputusan: Investor dan pelaku pasar membutuhkan alat atau analisis yang dapat membantu mereka dalam memahami potensi pergerakan harga saham untuk mendukung keputusan investasi yang lebih informasional.
- Identifikasi faktor berpengaruh: Belum jelas secara pasti faktor-faktor apa dari data historis harga saham (seperti harga pembukaan, tertinggi, terendah, penutupan, dan volume) yang paling berpengaruh terhadap pergerakan harga atau tren terkait waktu.

### Goals

Berdasarkan pernyataan masalah di atas dan analisis yang dilakukan dalam kode, berikut adalah tujuan proyek ini:

- Melakukan eksplorasi data historis harga saham: Menganalisis data historis harga saham untuk mengidentifikasi tren, pola, dan karakteristik distribusi data melalui teknik Exploratory Data Analysis (EDA), termasuk visualisasi distribusi fitur numerik, analisis tren data terkait waktu, dan analisis korelasi antar fitur.
- Mengidentifikasi potensi hubungan antar fitur: Mengidentifikasi bagaimana fitur-fitur numerik seperti harga pembukaan, penutupan, tertinggi, terendah, dan volume saling berhubungan dan berkorelasi.
- Mengembangkan model machine learning untuk prediksi: Membangun dan melatih model machine learning (seperti Random Forest dan Logistic Regression) menggunakan data historis untuk memprediksi pergerakan atau klasifikasi terkait waktu berdasarkan fitur-fitur yang tersedia.
- Mengevaluasi performa model prediktif: Mengevaluasi kinerja model yang dikembangkan menggunakan metrik yang relevan (seperti akurasi, precision, recall, dan F1-score) untuk menentukan model mana yang paling efektif dalam melakukan prediksi.
- Memberikan wawasan berbasis data: Menyajikan hasil analisis dan evaluasi model untuk memberikan wawasan berbasis data mengenai dinamika harga saham dan potensi prediksinya, yang dapat menjadi referensi awal untuk pengambilan keputusan investasi (meskipun ini bukan saran investasi finansial).

### Solution statements
1. Melakukan proses Exploratory Data Analysis (EDA) untuk memahami karakteristik data historis seperti harga pembukaan, penutupan, volume perdagangan, serta pola pergerakan nilai saham. Analisis korelasi dilakukan untuk mengidentifikasi fitur-fitur yang redundan, serta mengukur seberapa besar hubungan antar fitur terhadap target variabel yang akan diprediksi.
2. Menggunakan 2 model machine learning yaitu Random Forest dan Logistic Regression untuk membandingkan efektivitas dalam melakukan prediksi target berdasarkan data historis. Logistic Regression digunakan sebagai baseline model, sedangkan Random Forest digunakan sebagai model non-linear yang lebih kompleks untuk menguji peningkatan akurasi.
3. Melakukan hyperparameter tuning pada kedua algoritma untuk menemukan kombinasi parameter terbaik seperti n_estimators, max_depth, dan criterion pada Random Forest serta penalty, solver, dan C pada Logistic Regression guna meningkatkan performa model secara signifikan.
4. Menggunakan confusion matrix, accuracy, precision, recall, dan f1-score sebagai metrik evaluasi pada masing-masing model machine learning untuk menilai performa klasifikasi, dan menentukan model terbaik berdasarkan skor metrik tertinggi.
5. Menampilkan perbandingan performa model dalam bentuk visualisasi barplot, untuk memberikan gambaran yang lebih jelas terhadap akurasi masing-masing model, sehingga proses pengambilan keputusan terhadap model terbaik menjadi lebih terukur dan objektif.

## Data Understanding
### Dataset
Dataset yang digunakan diambil dari platform kaggle yang dapat diakses pada tautan berikut (https://www.kaggle.com/datasets/caesarmario/samsung-electronics-stock-historical-price) yang dipublikasikan oleh Mario Caesar. Kumpulan data ini berisi informasi lengkap history harga saham dari tahun 2019 sampai tahun 2025.
Infromasi dataset tersebut dapat dilihat pada gambar dibawah ini:

### Jumlah Baris dan Kolom

![image](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/Screenshot%20from%202025-06-04%2016-44-55.png)

Dari tabel diatas terdapat 1505 baris dan 7 kolom pada dataset.

![image](https://github.com/user-attachments/assets/9f5fe156-387a-482d-a53e-04e0dd44199b)

Dari hasil informasi terlihat bahwa :
- Terdapat 1 kolom dengan tipe object, yaitu : Date.
  - Kolom ini harusnya bertipe datetime karena mempresentasikan tanggal, bukan tipe data object yang digunakan untuk teks (string)
- Terdapat 5 kolom dengan tipe float64, yaitu : Open, High, Low, Close, dan Adj Close.
  - Tipe data sudah sesuai dengan isi dari kolomya.
- Terdapat 1 kolom dengan tipe int64, yaitu : volume.
  - Tipe data sudah sesuai dengan isi dari kolomya.

### Kondisi Data
#### Cek Missing Value
![image](https://github.com/user-attachments/assets/d1d213ca-2c57-446b-80b5-c4d2878552ed)

Dari hasil yang ditampilkan, data tidak ada yang bernilai kosong (null) pada setiap kolom dataset.

#### Cek Duplikasi Data
![image](https://github.com/user-attachments/assets/9d1472ca-6420-4c68-9701-e0eddc766dde)

Dari hasil yang ditampilkan, data tidak ada yang bernilai duplikat atau ganda pada setiap kolom dataset.

#### Cek Outlier
![image](https://github.com/user-attachments/assets/134e390e-d3df-478c-bea2-358b5a4a38bc)

Terlihat dari visualisasi boxplot diatas, fitur numerik dari kolom Volume memiliki outliers.

## Uraian seluruh fitur pada data
### Variabel pada dataset adalah sebagai berikut:
| Kolom       | Deskripsi                                                                         |
| ----------- | --------------------------------------------------------------------------------- |
| `Date`      | Tanggal pencatatan data saham                                                     |
| `Open`      | Harga pembukaan saham pada hari tersebut                                          |
| `High`      | Harga tertinggi yang dicapai pada hari tersebut                                   |
| `Low`       | Harga terendah yang dicapai pada hari tersebut                                    |
| `Close`     | Harga penutupan saham pada hari tersebut                                          |
| `Adj Close` | Harga penutupan yang telah disesuaikan (misalnya, untuk stock split atau dividen) |
| `Volume`    | Jumlah saham yang diperdagangkan pada hari tersebut                               |

### Exploratory Data Analysis - Univariate Analysis
#### Fitur Numerik

![image](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/download%20(3).png)

Berdasarkan grafik distribusi data dari numerical_feature yang ditampilkan, dapat dilihat bahwa :
1. Rentang nilai dari setiap variabel (Open, High, Low, Close) memiliki rentang nilai yang serupa, yaitu 40.000 hingga 90.000. Namun pada Adj_Close berkisar 30.000 hingga 80.000, sedangkan Volume berkisar 0 sampai 35.000.000.
2. Pada Adj_Close menunjukkan bahwa harga penutupan yang disesuaikan lebih stabil atau tidak mencapai nilai ekstrem seperti Close (90.000).
3. Pada Volume tidak ada rentang nilai yang jelas, tetapi jika dianggap dalam skala yang berbeda (misalnya jutaan), distribusi mungkin sangat miring (right-skewed) karena volume perdagangan seringkali memiliki outlier tinggi.

#### Fitur Datetime

![image](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/download%20(4).png)

Berdasarkan grafik dari jumlah data per tahun , dapat disimpulkan bahwa :
- 2019 terdapat 246 data
- 2020 terdapat 209 data
- 2021 terdapat 222 data
- 2022 terdapat 242 data
- 2023 terdapat 245 data
- 2024 terdapat 208 data
- 2025 terdapat 28 data

### Exploratory Data Analysis - Multivariate Analysis
#### Melihat korelasi variabel numerik dengan menggunakan Heatmap

![image](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/download%20(6).png)

Berdasarkan gambar diatas, terlihat bahwa :
1. Fitur Open, High, Low, Close, Adj Close punya korelasi mendekati 1, artinya mereka mengandung informasi yang hampir sama.
2. Volume perdagangan tidak berkorelasi langsung dengan perubahan harga saham.

#### visualisasi time series

![image](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/download%20(7).png)

Dari tahun 2018 hingga awal 2021, terlihat tren kenaikan harga saham yang signifikan, terutama lonjakan tajam di akhir 2020 hingga awal 2021. Setelah mencapai puncak di awal 2021 (~90.000), harga mulai mengalami fluktuasi dan penurunan bertahap.

## Data Preparation
Pada tahap ini kita akan melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahap persiapan data perlu dilakukan, yaitu :

### Konversi tipe data
![image](https://github.com/user-attachments/assets/2ab963c8-acca-4416-bffd-13e3e8236d65)

- Mengekstraksi Tahun: Baris kode `df['Year'] = df['Date'].dt.year` mengambil nilai tahun dari setiap entri di kolom Date dan menyimpannya ke dalam kolom baru bernama Year. Atribut .dt memungkinkan mengakses komponen-komponen datetime seperti tahun, bulan, hari, dll.

- Mengekstraksi Bulan: Baris kode `df['Month'] = df['Date'].dt.month` melakukan hal yang serupa, tetapi mengekstrak nilai bulan dari kolom Date dan menyimpannya ke dalam kolom baru bernama Month.

### Rename
![image](https://github.com/user-attachments/assets/e3626491-2469-4749-986a-abb1a4e69847)

Baris kode ini akan membuat DataFrame baru (yang kemudian ditugaskan kembali ke variabel df) di mana kolom yang sebelumnya bernama 'Adj Close' sekarang bernama 'Adj_Close'. Kolom-kolom lainnya akan tetap sama namanya.

### Hapus Outlier
Kita akan menggunakan metode IQR untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier.

![image](https://github.com/user-attachments/assets/99eec40c-9b41-4efa-a7af-42733285fb35)

Berikut hasilnya

![image](https://github.com/user-attachments/assets/7a0f1736-2f8a-4e6b-a693-ad0bbf899c0e)

Pada kolom 'Volume', dilakukan analisis outlier menggunakan metode Interquartile Range (IQR). Nilai Q1 (kuartil pertama) adalah 11.299.738, yang berarti 25% data memiliki nilai lebih kecil dari angka ini. Sementara itu, Q3 (kuartil ketiga) adalah 20.001.134, artinya 75% data memiliki nilai lebih kecil dari angka tersebut. Selisih antara Q3 dan Q1 disebut IQR, yang dalam hal ini sebesar 8.701.396.

Berdasarkan metode IQR, batas bawah untuk mendeteksi outlier adalah Q1 - 1.5 × IQR, yang menghasilkan nilai -1.752.356. Karena volume tidak mungkin bernilai negatif, tidak ditemukan outlier pada sisi bawah. Sedangkan batas atas ditentukan dari Q3 + 1.5 × IQR, yaitu 33.053.228. Setiap nilai volume yang melebihi batas atas ini dianggap sebagai outlier atas.

Dari hasil perhitungan, ditemukan sebanyak 66 baris data yang memiliki nilai volume melebihi batas atas tersebut. Dengan demikian, dapat disimpulkan bahwa terdapat 66 outlier pada kolom 'Volume', yang semuanya merupakan outlier atas.

### Drop Kolom
Menghapus kolom pada dataset yang tidak perlu digunakan dalam pemrosesan data yakni Date. Kolom ini akan dihapus menggunakan fungsi drop()

![image](https://github.com/user-attachments/assets/f835a0c6-441b-4dfb-a49f-5c0b5849c0c9)

Selnjutnya, kita akan mengecek informasi terbaru menggunakan fungsi info()

![image](https://github.com/user-attachments/assets/adc20a5d-59d2-4023-91ae-ea1ba02cf677)

Dari gambar informasi diatas kolom Date sudah berhasil dihapus.

### Train-Test-Split
Selanjutnya kita membagi data menjadi 2, yaitu Data training sebesar 80% untuk melatih model Data testing sebesar 20% untuk menguji model.

![image](https://github.com/user-attachments/assets/3a0d70dc-eb9e-4709-b0d7-2adce5e170b4)

Secara singkat, kedua baris kode ini melakukan pemisahan dataset Anda menjadi dua bagian:
x: Berisi fitur-fitur yang akan menjadi input bagi model machine learning (semua kolom kecuali 'Year').
y: Berisi target yang ingin diprediksi oleh model (kolom 'Year').

![image](https://github.com/user-attachments/assets/84108b19-767c-43c3-ac7c-705da880c43e)

Baris kode ini menggunakan fungsi train_test_split dari library scikit-learn, yang merupakan library populer untuk machine learning di Python. Fungsi ini digunakan untuk membagi dataset Anda menjadi subset pelatihan (training set) dan subset pengujian (testing set).

![image](https://github.com/user-attachments/assets/eafe2047-a69b-4716-931f-0eb65039ccd5)

Gambar diatas adalah hasil pembagian dataset, berikut merupakan penjelasannya :
- x_train: Fitur-fitur untuk subset pelatihan (80% data).
- x_test: Fitur-fitur untuk subset pengujian (20% data).
- y_train: Target untuk subset pelatihan (80% data, sesuai dengan x_train).
- y_test: Target untuk subset pengujian (20% data, sesuai dengan x_test).

## Model Development

Pada tahap ini, kita akan mengembangkan model machine learning dengan dua algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik.

### Random Forest
Cara kerja algoritma ini dengan cara menggabungkan hasil dari banyak pohon keputusan (decision trees). Ide dasarnya adalah membuat banyak pohon keputusan yang 'lemah' secara individual, dan kemudian menggabungkan prediksi mereka untuk mendapatkan prediksi yang lebih kuat dan stabil.

Langkah pertama kita melatih model dengan algoritma random forest dengan memanggil fungsi RandomForestClassifier. Parameter yang digunakan yaitu:
- `n_estimator`: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
- `max_depth`: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- `random_state`: digunakan untuk mengontrol random number generator yang digunakan.

![image](https://github.com/user-attachments/assets/5c0d3753-9273-481a-ab7e-fe054d11d6a5)

Berikut merupakan akurasi yang didapat oleh Random Forest

![image](https://github.com/user-attachments/assets/004925d3-f635-419e-a69e-705e88a61d4e)


### Logistic Regression

Logistic Regression pada submission ini memodelkan hubungan linier antara fitur-fitur harga/volume dan kemungkinan suatu data point (hari) termasuk dalam tahun tertentu, menggunakan fungsi softmax untuk mengubah skor linier menjadi probabilitas dan memilih tahun dengan probabilitas tertinggi sebagai prediksi.

Langkah kedua kita melatih model dengan algoritma logistic regression dengan memanggil fungsi LogisticRegression. Parameter yang digunakan yaitu:

`penalty='l2'` : jenis regularisasi yang digunakan. 'l2' berarti menggunakan L2 regularization (juga disebut Ridge), yang akan menambahkan penalti terhadap besar koefisien agar model tidak overfitting.

`C=1.0` : Ini adalah parameter regularisasi. C adalah kebalikan dari kekuatan regularisasi, jadi semakin kecil nilainya, semakin kuat regularisasinya.

`max_iter=1000` : Menentukan jumlah iterasi maksimum untuk solver konvergen. Default-nya adalah 100, tapi di sini ditingkatkan ke 1000 agar memastikan model bisa menemukan solusi optimal jika datanya besar atau rumit.

`class_weight='balanced'` : Berguna saat dataset tidak seimbang (misalnya, jumlah data kelas positif jauh lebih sedikit dari kelas negatif). Otomatis menghitung bobot kelas sebagai kebalikan dari frekuensi kelas di data pelatihan.

`solver='lbfgs'` : Ini adalah algoritma optimisasi yang digunakan untuk menemukan koefisien terbaik.

`'lbfgs'` : cocok untuk dataset kecil hingga menengah dan mendukung penalti 'l2'.

`random_state=42` : Untuk memastikan hasil yang reproducible (konsisten) di setiap kali eksekusi. Angka 42 bisa diganti apa saja, asalkan nilainya tetap untuk hasil yang konsisten.

![image](https://github.com/user-attachments/assets/2c69458d-f0fe-42c9-9632-e56271774904)

Berikut merupakan akurasi yang didapat oleh Logistic Regresion

![image](https://github.com/user-attachments/assets/84e92e1f-98e0-4afd-a2d3-922ba44ef033)

## Evaluation
### Menghitung metrik precision, recall, F1-score, dan support.

![image](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/Screenshot%20from%202025-06-04%2016-25-40.png)

Berdasarkan hasil evaluasi performa model, Random Forest menunjukkan kinerja yang secara signifikan lebih baik dibandingkan Logistic Regression. Model Random Forest menghasilkan akurasi sebesar 96%, dengan nilai precision, recall, dan f1-score yang tinggi dan konsisten di semua kelas, termasuk kelas dengan jumlah data yang sedikit. Sebaliknya, Logistic Regression hanya mencapai akurasi 80%, dengan performa yang cenderung menurun drastis pada beberapa kelas, terutama pada kelas minoritas. Oleh karena itu, dapat disimpulkan bahwa Random Forest lebih tepat digunakan untuk kasus klasifikasi ini karena mampu menangani kompleksitas data dengan lebih baik dan memberikan hasil yang lebih andal.

Berikut merupakan kesimpulan dari metrik-metrik yang ditampilkan pada hasil evaluasi performa

1. Random Forest
    - Memberikan hasil terbaik di seluruh metrik evaluasi: accuracy, precision, recall, dan F1-score.
    - Menunjukkan kinerja yang stabil pada semua kelas, termasuk kelas minoritas (2025) dengan F1-score yang tetap tinggi (0.92).
    - Cocok digunakan untuk dataset dengan distribusi kelas yang tidak seimbang, karena mampu mengenali pola dari kelas kecil maupun besar dengan baik.
    - Akurasi total mencapai 96%, yang menunjukkan performa sangat tinggi.
2. Logistic Regression
    - Memiliki akurasi lebih rendah (80%) dibandingkan Random Forest.
    - Kinerja model kurang stabil, dengan penurunan signifikan pada kelas minoritas (F1-score kelas 2025 hanya 0.45).
    - Lebih cocok untuk dataset yang seimbang, dan mungkin memerlukan penyesuaian (seperti oversampling/undersampling) jika digunakan pada data yang tidak seimbang.
    - Secara keseluruhan, performanya tidak sebaik Random Forest untuk kasus ini.

### Melihat perbandingan akurasi model dengan grafik barplot

![image](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/download%20(2).png)

Dari grafik barplot yang ditampilkan, terlihat bahwa model menggunakan algoritma Random Forest lebih tinggi akurasinya yakni 0.9571 dibandingkan dengan model yakni Naive Bayes = 0.7964 untuk memprediksi harga saham setiap tahun.

## Referensi
1. Permana, N. A., & Bunyamin, H. (2024). Perbandingan Logistic Regression dengan Random Forest dalam Memprediksi Sentimen Pada IMDb Moview Review. Jurnal STRATEGI-Jurnal Maranatha, 6(2), 391-399.
2. Febriyanti, N. R., Kusrini, K., & Hartanto, A. D. (2025). Analisis Perbandingan Algoritma SVM, Random Forest dan Logistic Regression untuk Prediksi Stunting Balita. Edumatic: Jurnal Pendidikan Informatika, 9(1), 149-158.
