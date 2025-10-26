from flask import Flask, render_template, request, jsonify
import os
from ml_model.model import extract_face_from_video, verify_faces

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Extract face from video
    video_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video_face.jpg')
    face_found = extract_face_from_video(video_path, video_face_path)

    if not face_found:
        return jsonify({"status": "fail", "message": "❌ No face detected in the video."})

    # Receive webcam image from JS
    webcam_file = request.files['webcam']
    webcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_face.jpg')
    webcam_file.save(webcam_path)

    result = verify_faces(video_face_path, webcam_path)

    if result:
        return jsonify({"status": "success", "message": "✅ Uploaded"})
    else:
        return jsonify({"status": "fail", "message": "❌ Only the person in the video can upload the video"})

if __name__ == "__main__":
    app.run(debug=True, port=5002)


