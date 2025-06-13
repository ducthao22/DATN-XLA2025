
import cv2
import numpy as np
import os
import math
import time
import cvzone
import smtplib
import ssl
from email.message import EmailMessage
import requests
from ultralytics import YOLO
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Cấu hình Email và Telegram
email_sender = 'tdtofficial10@gmail.com'
email_password = 'cgsirfskhmyrflvd'
email_receiver = 'thanducthao2002@gmail.com'
TELEGRAM_TOKEN = '7517823069:AAGLwSGDHp3G2gpNTwYdiopt2Q6f7JAA4XY'
TELEGRAM_CHAT_ID = '1063446083'

# Cấu hình còi báo
BUZZER_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# Gửi email
def send_email_with_image(subject, body, image_path):
    try:
        em = EmailMessage()
        em['From'] = email_sender
        em['To'] = email_receiver
        em['Subject'] = subject
        em.set_content(body)

        with open(image_path, 'rb') as img:
            em.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ssl.create_default_context()) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(em)
            print("Email cảnh báo đã gửi!")
    except Exception as e:
        print(f"Lỗi email: {str(e)}")

# Gửi Telegram
def send_telegram_alert(message, image_path):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(image_path, 'rb') as photo:
            response = requests.post(
                url,
                files={'photo': photo},
                data={'chat_id': TELEGRAM_CHAT_ID, 'caption': message}
            )
            if response.status_code == 200:
                print("Đã gửi cảnh báo Telegram!")
            else:
                print(f"Lỗi Telegram: {response.text}")
    except Exception as e:
        print(f"Lỗi gửi Telegram: {str(e)}")

# Load mô hình nhận diện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['Ronaldo', 'Thao', 'Putin', 'Messi', 'D. Trump', 'W']

# Load mô hình YOLO phát hiện lửa
model = YOLO('fire.pt')
classnames = ['fire']

# Khởi động camera Pi
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

minW = 0.1 * 640
minH = 0.1 * 480
last_alert = {'fire': 0, 'unknown': 0}
COOLDOWN = 50
buzzer_active = False

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nhận diện khuôn mặt
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 50:
                name = names[id]
            else:
                name = "unknown"

            cv2.putText(frame, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"{round(100 - confidence)}%", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            # Cảnh báo người lạ
            if name == "unknown" and (time.time() - last_alert['unknown']) > COOLDOWN:
                img_path = 'stranger_alert.jpg'
                cv2.imwrite(img_path, frame)
                send_email_with_image("CẢNH BÁO NGƯỜI LẠ", "Phát hiện người không xác định!", img_path)
                send_telegram_alert("🚨 CẢNH BÁO NGƯỜI LẠ!\nTại: Cửa chính", img_path)
                last_alert['unknown'] = time.time()

        # Phát hiện lửa
        fire_detected = False
        result = model(frame, stream=True)
        for info in result:
            for box in info.boxes:
                confidence = math.ceil(box.conf[0] * 100)
                Class = int(box.cls[0])

                if confidence > 80:
                    fire_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 30],
                                       scale=1.5, thickness=2)

        # Cảnh báo lửa
        if fire_detected:
            if not buzzer_active:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                buzzer_active = True
                print("🔥 Buzzer ON")

            if (time.time() - last_alert['fire']) > COOLDOWN:
                img_path = 'fire_alert.jpg'
                cv2.imwrite(img_path, frame)
                send_email_with_image("CẢNH BÁO CHÁY", "Phát hiện lửa!", img_path)
                send_telegram_alert("🔥 CẢNH BÁO CHÁY!\nTại: Phòng 301 - Tòa A", img_path)
                last_alert['fire'] = time.time()
        else:
            if buzzer_active:
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                buzzer_active = False

        # Hiển thị kết quả
        cv2.imshow('Nhan Dien Khuon Mat & Fire Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()
    picam2.close()
    cv2.destroyAllWindows()
