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
