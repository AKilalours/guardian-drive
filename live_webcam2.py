import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')
eye_tree = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Capture 5 frames and pick best
best_eyes = 0
best_frame = None
best_ear = 0.30

for _ in range(10):
    ret, frame = cap.read()
    if not ret: continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improve contrast
    
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3,
        minSize=(100,100))
    
    if len(faces) > 0:
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        # Look for eyes in upper half of face only
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05,
            minNeighbors=3, minSize=(20,20))
        
        if len(eyes) < 2:
            eyes2 = eye_tree.detectMultiScale(
                roi_gray, scaleFactor=1.05,
                minNeighbors=3, minSize=(20,20))
            if len(eyes2) > len(eyes):
                eyes = eyes2
        
        n_eyes = len(eyes)
        if n_eyes >= best_eyes:
            best_eyes = n_eyes
            best_frame = frame.copy()
            if n_eyes >= 2:
                h_vals = [e[3] for e in eyes[:2]]
                w_vals = [e[2] for e in eyes[:2]]
                best_ear = min(0.35, np.mean(h_vals)/np.mean(w_vals)*0.55)
            elif n_eyes == 1:
                best_ear = 0.22
            else:
                best_ear = 0.10
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        
        cv2.putText(frame, f"Eyes: {n_eyes}  EAR~{best_ear:.3f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,255), 2)

cap.release()

if best_frame is not None:
    cv2.imwrite("/tmp/webcam_test2.jpg", best_frame)
    print(f"Best result: {best_eyes} eyes detected, EAR~{best_ear:.3f}")
    print("Saved: /tmp/webcam_test2.jpg")
    open("/tmp/webcam_test2.jpg")
else:
    print("No face detected")
