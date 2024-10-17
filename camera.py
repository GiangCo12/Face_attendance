import cv2
import os

# Khởi tạo webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Kiểm tra đường dẫn tới haarcascade_frontalface_default.xml
cascade_path = '/Users/user/Downloads/source/haarcascade_frontalface_default.xml'  # Đảm bảo đường dẫn đúng
if not os.path.exists(cascade_path):
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit(1)

face_detector = cv2.CascadeClassifier(cascade_path)

# Nhập ID cho người dùng
face_id = input('\n enter user id end press <return> ==>  ')

# Đường dẫn lưu ảnh
output_dir = 'dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0

while True:
    ret, img = cam.read()
    
    if not ret:  # Kiểm tra xem có đọc được khung hình hay không
        print("Failed to capture image. Exiting...")
        break

    img = cv2.flip(img, 1)  # Lật ảnh video theo chiều dọc
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kiểm tra xem gray có rỗng không trước khi sử dụng
    if gray is None or gray.size == 0:
        print("Captured an empty frame. Exiting...")
        break

    # Phát hiện khuôn mặt
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # Lưu ảnh vào thư mục chỉ định
        image_path = os.path.join(output_dir, f"User.{face_id}.{count}.jpg")
        cv2.imwrite(image_path, gray[y:y + h, x:x + w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 200: 
         break

# Dọn dẹp
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
