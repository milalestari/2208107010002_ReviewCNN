from google.colab import files
from keras.models import load_model
from PIL import Image
import numpy as np

# Fungsi untuk memuat dan memproses gambar
def load_and_prepare_image(file_path):
    img = Image.open(file_path)
    img = img.resize((32, 32))  # Sesuaikan dengan dimensi yang model Anda harapkan
    img = np.array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension
    return img

# Nama-nama kelas untuk dataset CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Muat model
#model = load_model('path_to_your_model.h5')  # Ganti dengan path ke model yang disimpan

# Mengunggah file gambar
uploaded = files.upload()

# Memproses gambar dan membuat prediksi
for filename in uploaded.keys():
    img = load_and_prepare_image(filename)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Dapatkan indeks kelas
    predicted_class_name = class_names[predicted_class_index]  # Dapatkan nama kelas
    print(f'File: {filename}, Predicted Class Index: {predicted_class_index}, Predicted Class Name: {predicted_class_name}')
