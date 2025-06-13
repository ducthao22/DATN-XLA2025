
from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Khởi tạo camera Pi
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(2)  # Chờ camera ổn định

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Nhap id khuon mat <return> ==> ')

print("\n [INFO] Khoi tao Camera ...")
count = 0

while True:
    img = picam2.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Nhấn ESC để thoát
        break
    elif count >= 20:  # Chụp 20 ảnh là dừng
        break

print("\n [INFO] Thoát")
cv2.destroyAllWindows()
