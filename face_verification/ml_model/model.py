import face_recognition
import cv2
import os

cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise ValueError(f"Failed to load Haar cascade xml from {cascade_path}")

def extract_face_from_video(video_path, save_path):
    """Extract first face from video and save as image"""
    video = cv2.VideoCapture(video_path)
    face_detected = False

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.imwrite(save_path, frame[y:y+h, x:x+w])
            face_detected = True
            break

    video.release()
    return face_detected

def verify_faces(video_face_path, webcam_face_path):
    video_face = face_recognition.load_image_file(video_face_path)
    webcam_face = face_recognition.load_image_file(webcam_face_path)

    video_enc = face_recognition.face_encodings(video_face)
    webcam_enc = face_recognition.face_encodings(webcam_face)

    if len(video_enc) == 0 or len(webcam_enc) == 0:
        return False  # No face detected

    result = face_recognition.compare_faces([video_enc[0]], webcam_enc[0])
    return result[0]
