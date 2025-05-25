from datetime import datetime
from firebase_config import db

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    attendance_ref = db.collection("attendance").document(date_str)
    doc = attendance_ref.get()

    if doc.exists:
        data = doc.to_dict()
        if name not in data:
            data[name] = time_str
            attendance_ref.set(data)
    else:
        attendance_ref.set({name: time_str})