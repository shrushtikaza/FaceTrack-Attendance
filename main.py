import cv2
import numpy as np
import face_recognition
from encoding import load_registered_users, register_new_user
from attendance import mark_attendance

encodeListKnown, classNames = load_registered_users()

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam for attendance... (Press 'r' to register new face, 'q' to quit)")

while True:
    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to read from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceloc_curr = face_recognition.face_locations(imgS)
    encode_curr = face_recognition.face_encodings(imgS, faceloc_curr)

    for encode_face, faceloc in zip(encode_curr, faceloc_curr):
        matches = face_recognition.compare_faces(encodeListKnown, encode_face)
        face_dist = face_recognition.face_distance(encodeListKnown, encode_face)

        if len(face_dist) > 0:
            matchIndex = np.argmin(face_dist)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                mark_attendance(name)

    cv2.imshow("Attendance Webcam", img)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        print("[INFO] Registration Mode Activated")
        name = input("Enter name for registration: ").strip()
        if name:
            register_new_user(name, img)
            encodeListKnown, classNames = load_registered_users()
            print(f"[INFO] Registered and refreshed data for: {name}")

cap.release()
cv2.destroyAllWindows()
