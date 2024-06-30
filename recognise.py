import cv2
import numpy as np
from keras.models import load_model
import json
import os
import base64

# Ukuran gambar
image_x, image_y = 64, 64

# Memuat model klasifikasi
try:
    classifier = load_model('Trained_model10class.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Fungsi untuk melakukan prediksi kelas dan menghitung kepercayaan
def predictor():
    from keras.preprocessing import image
    try:
        test_image = image.load_img('1.png', target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Normalisasi data
        test_image = np.expand_dims(test_image, axis=0)
        print(f"Input to model: {test_image.shape}")  # Debug: tampilkan shape dari input
        result = classifier.predict(test_image)
        print(f"Model prediction result: {result}")  # Debug: tampilkan hasil prediksi
        predicted_class_index = np.argmax(result)
        predicted_class = ""
        if predicted_class_index == 0:
            predicted_class = 'dan'
        elif predicted_class_index == 1:
            predicted_class = 'hai'
        elif predicted_class_index == 2:
            predicted_class = 'i love you'
        elif predicted_class_index == 3:
            predicted_class = 'kami'
        elif predicted_class_index == 4:
            predicted_class = 'kasih'
        elif predicted_class_index == 5:
            predicted_class = 'kita'
        elif predicted_class_index == 6:
            predicted_class = 'maaf'
        elif predicted_class_index == 7:
            predicted_class = 'saya'
        elif predicted_class_index == 8:
            predicted_class = 'semangat'
        elif predicted_class_index == 9:
            predicted_class = 'takut'
        confidence = np.max(result) * 100
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

# Fungsi untuk menyimpan hasil prediksi ke dalam file JSON
def save_to_json(predicted_class, confidence, img_name):
    try:
        with open(img_name, "rb") as img_file:
            img_str = base64.b64encode(img_file.read()).decode('utf-8')

        data = {
            'prediction': predicted_class,
            'confidence': confidence,
            'image': img_str
        }

        # Mengecek apakah file sudah ada
        if os.path.exists('results.json'):
            with open('results.json', 'r') as f:
                results = json.load(f)
                if isinstance(results, dict):  # Convert to list if it is a single dictionary
                    results = [results]
        else:
            results = []

        results.append(data)

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving to JSON: {e}")

# Membuka kamera
cam = cv2.VideoCapture(2)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    imcrop = img[102:298, 427:623]
    gray = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
    
    # Menggunakan thresholding Otsu untuk konversi hitam putih
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)

    # Simpan gambar untuk prediksi
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))

    # Tambahkan langkah untuk menyimpan gambar yang di-crop
    cv2.imwrite("cropped_image.png", imcrop)
    
    # Lakukan prediksi dan hitung akurasi
    predicted_class, confidence = predictor()
    img_text = f"{predicted_class}, Conf: {confidence:.2f}%"
    
    # Simpan hasil prediksi ke dalam file JSON jika akurasi di atas 97%
    if confidence >= 60:
        save_to_json(predicted_class, confidence, img_name)
    
    # Tampilkan hasil prediksi dan akurasi di layar
    cv2.putText(frame, img_text, (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("test", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
