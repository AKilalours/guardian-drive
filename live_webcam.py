"""
Standalone webcam test using OpenCV only (no MediaPipe).
Detects eye regions and estimates EAR using Haar cascades.
"""
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

def estimate_ear_opencv(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return 0.30, frame, "No face detected"
    
    x,y,w,h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    if len(eyes) == 0:
        # No eyes detected = likely closed
        ear = 0.10
        status = "Eyes not detected (closed?)"
    elif len(eyes) == 1:
        ear = 0.20
        status = "One eye detected"
    else:
        # Both eyes detected = open
        # Estimate EAR from eye height/width ratio
        eye_h = np.mean([e[3] for e in eyes])
        eye_w = np.mean([e[2] for e in eyes])
        ear = min(0.35, eye_h/eye_w * 0.5)
        status = f"Both eyes open (EAR~{ear:.3f})"
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame, (x+ex,y+ey),
                         (x+ex+ew,y+ey+eh), (0,0,255), 2)
    
    color = (0,0,255) if ear<0.18 else (0,165,255) if ear<0.22 else (0,255,0)
    cv2.putText(frame, f"EAR: {ear:.3f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, status, (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    return ear, frame, status

# Test
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    ear, annotated, status = estimate_ear_opencv(frame)
    print(f"OpenCV EAR detection: {status}")
    print(f"EAR estimate: {ear:.3f}")
    # Save test frame
    cv2.imwrite("/tmp/webcam_test.jpg", annotated)
    print("Test frame saved: /tmp/webcam_test.jpg")
else:
    print("Webcam failed")
