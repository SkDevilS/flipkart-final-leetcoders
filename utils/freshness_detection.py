import cv2
from models.freshness_model import FreshnessDetectionModel

model = FreshnessDetectionModel()

def detect_freshness():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        freshness, name = model.detect(frame)
        cv2.putText(frame, f"{name}: {freshness}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
