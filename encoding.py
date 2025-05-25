import face_recognition
import numpy as np
from firebase_config import db
import cv2
import base64
import json

def load_registered_users():
    users_ref = db.collection("users")
    docs = users_ref.stream()
    encodeListKnown = []
    classNames = []

    for doc in docs:
        data = doc.to_dict()
        name = data["name"]
        encoding_str = data["encoding"]
        encoding = np.array(json.loads(encoding_str))
        encodeListKnown.append(encoding)
        classNames.append(name)
    
    return encodeListKnown, classNames

def register_new_user(name, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(img_rgb)
    if not face_locs:
        print("[ERROR] No face detected!")
        return

    face_enc = face_recognition.face_encodings(img_rgb, face_locs)[0]
    enc_list = face_enc.tolist()
    db.collection("users").document(name).set({
        "name": name,
        "encoding": json.dumps(enc_list)
    })
    print(f"[INFO] Registered new user: {name}")