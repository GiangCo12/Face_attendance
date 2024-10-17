import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

# Path để lưu mô hình sau khi training
trainer_path = 'trainer'
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)  # Tạo thư mục trainer nếu chưa có

# Khởi tạo LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Hàm để lấy ảnh và nhãn dữ liệu
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu model vào file trainer.yml
recognizer.write(os.path.join(trainer_path, 'trainer.yml'))

# Thông báo số lượng khuôn mặt đã huấn luyện
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
