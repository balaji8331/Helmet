import os
import numpy as np
import cv2
import pygame
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, Response

app = Flask(__name__)

# Initialize helmet detection model and video capture
network = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
model = keras.models.load_model('helmet.h5')
vid = cv2.VideoCapture(0)

# Initialize pygame for sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(r"C:\Users\manoj\Music\emergency-alarm-with-reverb-29431.mp3") 

# Function to simulate integration with other safety systems
def integrate_with_safety_systems():
    print("Helmet detected! Activating other safety systems...")

    # Simulate activation of safety items
    print("1. Emergency Alarm Activated")
    print("2. Access Control System Activated")
    print("3. Alerting Security Personnel")

    # Add your specific logic here to interact with other safety components
    # For example, you could call APIs, send signals, or trigger events in other safety-related systems

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        ret, frame = vid.read()
        if ret:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1, (300, 300), (104.0, 177.0, 123.0))
            network.setInput(blob)
            detections = network.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype(int)

                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX - 100, startY - 100), (endX + 50, endY + 50), (0, 0, 255), 2)
                    temp = frame[startY - 100:endY + 100, startX - 100:endX + 100]
                    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    temp = cv2.resize(temp, (224, 224))
                    temp = preprocess_input(temp)
                    temp = np.expand_dims(temp, axis=0)
                    pred_val = model.predict(temp)
                    print(pred_val)
                    pred_val = np.ravel(pred_val).item()

                    if pred_val < 0.7:
                        text = 'NO-HELMET' + str(pred_val)
                        cv2.rectangle(frame, (startX - 100, startY - 100), (endX + 50, endY + 50), (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        # Sound the alarm
                        alert_sound.play()

                        # Simulate integration with other safety systems
                        integrate_with_safety_systems()
                    else:
                        text = 'HELMET' + str(pred_val)
                        cv2.rectangle(frame, (startX - 100, startY - 100), (endX + 50, endY + 50), (0, 255, 0), 2)
                        cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Convert the frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
