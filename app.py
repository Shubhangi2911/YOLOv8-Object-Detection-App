from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
from datetime import datetime
import shutil
import cv2

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/uploads'
DETECT_FOLDER = 'static/detected'
RUNS_FOLDER = os.path.join('runs', 'detect')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)
os.makedirs(RUNS_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- IMAGE DETECTION ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('predict.html', error='⚠️ Please upload an image.')

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        temp_output = os.path.join(RUNS_FOLDER, 'predict')
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)

        # Run YOLOv8 detection
        model.predict(source=upload_path, save=True, project=RUNS_FOLDER, name='predict')

        detected_file = None
        for f in os.listdir(temp_output):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                detected_file = os.path.join(temp_output, f)
                break

        if not detected_file:
            return render_template('predict.html', error='❌ Detection failed.', uploaded_image=f'uploads/{filename}')

        detected_filename = 'detected_' + filename
        detected_output_path = os.path.join(DETECT_FOLDER, detected_filename)
        shutil.copy(detected_file, detected_output_path)

        return render_template(
            'predict.html',
            uploaded_image=f'uploads/{filename}',
            detected_image=f'detected/{detected_filename}'
        )

    return render_template('predict.html')

# ---------------- LIVE DETECTION ----------------
def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("❌ Could not open webcam")

    class_names = model.names  # class labels (like 'person', 'car', etc.)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run YOLOv8 on the current frame
        results = model.predict(frame, verbose=False)  # process single frame

        # Access detection boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordinates
                conf = float(box.conf[0])               # confidence
                cls = int(box.cls[0])                   # class ID
                label = f"{class_names[cls]} {conf:.2f}"

                # Draw rectangle + label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream to client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- MAIN ----------------
if __name__ == '__main__':
    app.run(debug=True)
