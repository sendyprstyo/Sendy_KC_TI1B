import cv2
import numpy as np

# Memuat model deteksi wajah dari OpenCV
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi kamera (0 untuk kamera internal, 1 untuk kamera eksternal)
camera = cv2.VideoCapture(0)

# Mengecek apakah kamera berhasil dibuka
if not camera.isOpened():
    print("Error: Kamera tidak terdeteksi!")
    exit()
else:
    print("Kamera berhasil dibuka.")

# Loop untuk menangkap video
while True: 
    ret, img = camera.read()
    if not ret:
        print("Error: Gagal mengambil frame!")
        break

    # Konversi ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = faceDetect.detectMultiScale(gray, 1.1, 5)

    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menampilkan hasil deteksi wajah
    cv2.imshow("Face Detection", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup jendela
camera.release()
cv2.destroyAllWindows()
