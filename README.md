# Review: Deep Learning dan Convolutional Neural Networks (CNN)

1. Pengantar CNN :
CNN adalah salah satu teknik deep learning yang dikembangkan untuk menangani data visual. Pendekatan ini memungkinkan komputer untuk mengenali objek secara otomatis, seperti pengenalan wajah di media sosial atau sistem keamanan berbasis CCTV. Dibandingkan dengan pendahulunya, Artificial Neural Networks (ANN), CNN lebih efektif untuk data visual karena kemampuan spesifiknya dalam mendeteksi fitur dalam gambar.

2. Kontribusi Pionir Deep Learning
CNN dikembangkan oleh tokoh-tokoh penting seperti Yann LeCun, Geoffrey Hinton, dan Yoshua Bengio, yang memenangkan Turing Award pada 2018. Ketiganya berperan besar dalam mempopulerkan dan mengembangkan metode ini, hingga menjadi tulang punggung berbagai aplikasi AI modern.

3. Cara Kerja CNN
CNN bekerja dengan meniru cara manusia mengenali objek:
- Komputer mengonversi gambar menjadi kumpulan angka (pixel) dengan nilai antara 0 hingga 255.
- Dalam gambar berwarna, nilai pixel dikelompokkan menjadi tiga dimensi (RGB - Red, Green, Blue).
- Dengan pengolahan berbasis angka, CNN dapat membaca pola visual melalui beberapa tahapan utama.

4. Tahapan Utama CNN
- Convolution
Menggunakan feature detector (juga disebut kernel/filter) untuk mengekstrak fitur dari gambar asli. Proses ini menghasilkan feature map, yaitu representasi gambar dengan ukuran lebih kecil yang hanya memuat informasi penting.
- ReLU (Rectified Linear Unit)
Fungsi aktivasi yang menghilangkan nilai negatif dari data, menjaga non-linearitas, sehingga membantu algoritma lebih cepat mencapai solusi optimal.
- Pooling
Mengurangi dimensi data lebih lanjut dengan menyaring fitur penting, tanpa kehilangan informasi inti. Misalnya, Max Pooling memilih nilai maksimum dari suatu area.
- Flattening dan Full Connection
Data yang telah disaring diubah menjadi bentuk vektor (1 dimensi) untuk dihubungkan ke lapisan fully connected. Proses ini menghasilkan prediksi akhir, seperti pengenalan objek dengan probabilitas tertentu.

5. Penerapan CNN
CNN digunakan dalam berbagai aplikasi nyata, termasuk:
- Media sosial: Pengenalan wajah otomatis (contoh: Facebook).
- Keamanan: Sistem pengawasan CCTV yang mengenali wajah atau plat kendaraan.
- Spionase dan teknologi futuristik: Identifikasi objek menggunakan perangkat seperti kontak lensa pintar.
  
6. Tantangan dan Kelebihan
- Kelebihan: CNN unggul dalam memproses data visual berukuran besar dengan efisiensi tinggi. Model ini dirancang untuk mendeteksi fitur secara hierarkis, dari pola sederhana hingga kompleks.
- Tantangan: Akurasi CNN sangat bergantung pada kualitas dataset untuk training. Misalnya, jika dataset tidak representatif, model dapat memberikan prediksi keliru.

7. Convolutional Neural Networks (CNN) dan Max Pooling
CNN bekerja dengan mengekstraksi fitur gambar melalui beberapa lapisan proses, termasuk convolution, pooling, dan flattening, untuk menghasilkan prediksi. Fokus utama pembahasan adalah max pooling, yaitu:
- Definisi: Teknik yang memilih nilai maksimum dalam area kecil (biasanya 2×2 atau 3×3) dari feature map hasil lapisan convolution.
- Tujuan: Menangkap fitur paling menonjol sambil menjaga invariansi posisi (gambar tetap dikenali meskipun diputar, dikecilkan, atau pencahayaan berbeda).
- Manfaat:
a. Mengurangi dimensi data, sehingga mempercepat komputasi.
b. Meningkatkan kemampuan generalisasi model.
c. Ilustrasi dengan gambar kucing menunjukkan bahwa max pooling membantu model mengenali pola dasar kucing, terlepas dari variasi posisi atau bentuknya.

2. Flattening
- Definisi: Proses meratakan data dari pooled feature maps menjadi bentuk vektor tunggal.
- Tujuan: Mengubah data dari bentuk matriks menjadi input yang bisa diproses oleh lapisan fully connected dalam Artificial Neural Networks (ANN).
- Manfaat: Mempermudah integrasi hasil ekstraksi fitur ke dalam tahap klasifikasi akhir.
  
3. Fully Connected Layers (ANN Integration)
Pada tahap ini, hasil flattening dimasukkan ke dalam jaringan ANN. Semua node di lapisan ini saling terhubung, dan setiap node mempengaruhi prediksi akhir berdasarkan bobotnya. Proses melibatkan:
- Forward Propagation: Menghitung nilai output berdasarkan input.
- Back Propagation: Menyesuaikan bobot berdasarkan kesalahan prediksi untuk meningkatkan akurasi.
- Prediksi akhir menggunakan fungsi softmax, menghasilkan probabilitas untuk setiap kategori (misalnya, anjing atau kucing).

4. Evaluasi dan Loss Function
- Evaluasi Model: CNN menggunakan cross-entropy loss function untuk menilai seberapa jauh prediksi dari hasil yang diharapkan.
- Output: Probabilitas untuk masing-masing kategori, di mana totalnya adalah 1.

https://github.com/milalestari/2208107010002_ReviewCNN/blob/main/2208107010002_poin2-1.py
Kode di atas membangun dan melatih model Convolutional Neural Network (CNN) menggunakan Keras untuk klasifikasi biner, misalnya untuk membedakan dua kategori gambar seperti kucing dan anjing. Pertama, model diinisialisasi dengan menggunakan Sequential dan menambahkan lapisan-lapisan CNN, termasuk lapisan konvolusi (Conv2D) untuk mengekstraksi fitur dari gambar, lapisan pooling (MaxPooling2D) untuk mengurangi dimensi data, lapisan flatten (Flatten) untuk mengubah data menjadi vektor 1D, dan lapisan fully connected (Dense) untuk klasifikasi akhir. Model dikompilasi menggunakan optimizer Adam dengan fungsi loss binary_crossentropy dan metrik akurasi. Data gambar dari direktori diproses menggunakan ImageDataGenerator, di mana augmentasi diterapkan pada training set (rescaling, shear, zoom, horizontal flip), sedangkan test set hanya dinormalisasi. Data ini kemudian dimuat dengan flow_from_directory dan disiapkan untuk pelatihan. Model dilatih menggunakan metode fit dengan parameter steps_per_epoch, jumlah epoch, validasi menggunakan test set, dan validation_steps. Pelatihan ini memungkinkan model mempelajari fitur dari gambar untuk memprediksi kelasnya dengan akurasi tinggi.


Kode di atas digunakan untuk menguji performa model CNN yang telah dilatih sebelumnya pada kumpulan gambar test set, khususnya gambar-gambar anjing dan kucing. Prosesnya dilakukan secara iteratif untuk 1000 gambar dalam direktori dataset/test_set/dogs/, dengan nama file berbentuk dog.<index>.jpg. Setiap gambar dimuat menggunakan fungsi image.load_img dengan ukuran yang disesuaikan menjadi 128x128 piksel agar sesuai dengan dimensi input model. Gambar kemudian diubah menjadi array menggunakan image.img_to_array dan dimensi array diperluas menggunakan np.expand_dims untuk memenuhi format input model. Model memprediksi gambar menggunakan metode predict, menghasilkan probabilitas untuk setiap kelas (0 untuk kucing dan 1 untuk anjing). Hasil prediksi dibandingkan dengan indeks kelas yang terdapat pada training set (training_set.class_indices) untuk menentukan jenis gambar. Prediksi dihitung dan dicatat dalam variabel count_cat (untuk kucing) dan count_dog (untuk anjing). Setelah iterasi selesai, jumlah gambar kucing dan anjing yang terdeteksi dicetak ke konsol untuk mengevaluasi hasil prediksi model.


Kode di atas membangun dan melatih model Convolutional Neural Network (CNN) menggunakan dataset CIFAR-10, yang berisi gambar berwarna 32x32 piksel dalam 10 kelas (seperti pesawat, mobil, burung, dll). Data CIFAR-10 diimpor, dibagi menjadi training dan test set, kemudian dinormalisasi dengan membagi nilai piksel dengan 255. Label data diubah menjadi bentuk kategorikal menggunakan to_categorical untuk memungkinkan klasifikasi multi-kelas.
Model CNN dibangun menggunakan API Keras dengan beberapa lapisan: tiga lapisan konvolusi dengan jumlah filter yang meningkat (32, 64, 128) untuk mengekstraksi fitur gambar, masing-masing diikuti oleh lapisan pooling untuk mengurangi dimensi data. Setelah lapisan konvolusi, hasilnya dipipihkan (Flatten) agar sesuai dengan lapisan dense. Lapisan dense dengan 128 neuron digunakan untuk memproses fitur yang diekstraksi, dengan dropout (50%) untuk mengurangi overfitting. Lapisan terakhir memiliki 10 neuron dengan aktivasi softmax untuk memprediksi probabilitas dari 10 kelas.
Model dikompilasi menggunakan optimizer Adam, fungsi loss categorical_crossentropy, dan metrik akurasi. Model dilatih pada data latih selama 10 epoch dengan ukuran batch 64, menggunakan data uji untuk validasi selama pelatihan. Setelah pelatihan selesai, model dievaluasi pada data uji untuk menghitung akurasi, yang dicetak ke konsol untuk menilai kinerja model pada data yang belum pernah dilihat.


Kode di atas digunakan untuk memprediksi kelas sebuah gambar berdasarkan model CNN yang telah dilatih sebelumnya pada dataset CIFAR-10. Proses dimulai dengan memuat beberapa library, termasuk PIL untuk memproses gambar, numpy untuk manipulasi data, dan keras untuk memuat model yang telah disimpan sebelumnya. Fungsi load_and_prepare_image dibuat untuk membaca gambar dari file, mengubah ukurannya menjadi 32x32 piksel (sesuai dimensi input model), menormalisasi piksel dengan membagi nilai piksel dengan 255, dan menambahkan dimensi batch untuk kompatibilitas dengan model.
Nama-nama kelas CIFAR-10 didefinisikan dalam daftar class_names, seperti "airplane", "automobile", hingga "truck". Kemudian, model diload menggunakan load_model (dengan asumsi path model disediakan). Gambar diunggah oleh pengguna menggunakan fitur files.upload() dari Google Colab, dan untuk setiap gambar yang diunggah, fungsi load_and_prepare_image dipanggil untuk memprosesnya.
Setelah gambar siap, model membuat prediksi dengan memproses gambar. Output prediksi adalah probabilitas untuk setiap kelas, dan indeks kelas dengan probabilitas tertinggi diambil menggunakan np.argmax. Indeks ini digunakan untuk mengambil nama kelas dari class_names. Akhirnya, nama file gambar, indeks kelas, dan nama kelas yang diprediksi ditampilkan untuk setiap gambar yang diunggah.
