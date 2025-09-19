"""
Mock Camera Version for Server Deployment
This version simulates camera input for demonstration purposes
"""
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2, os, numpy as np
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

USER_DIR = "users"
if not os.path.exists(USER_DIR):
    os.makedirs(USER_DIR)

ATTENDANCE_DIR = "attendance_collections"
if not os.path.exists(ATTENDANCE_DIR):
    os.makedirs(ATTENDANCE_DIR)

# Mock camera globals
capture_progress = {'current': 0, 'total': 300, 'status': 'idle'}
attendance_data = []
global_face_recognizer = None
user_labels = {}
label_names = {}

def create_mock_face_image():
    """Create a mock face image for demonstration"""
    # Create a simple face-like image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Draw a simple face shape
    center = (320, 240)
    cv2.circle(img, center, 100, (255, 200, 150), -1)  # Face circle
    cv2.circle(img, (290, 210), 15, (0, 0, 0), -1)     # Left eye
    cv2.circle(img, (350, 210), 15, (0, 0, 0), -1)     # Right eye
    cv2.ellipse(img, (320, 270), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    # Add green rectangle to simulate face detection
    cv2.rectangle(img, (220, 140), (420, 340), (0, 255, 0), 2)
    
    # Add mock progress text
    progress_text = f"MOCK CAMERA - Demo Mode"
    cv2.putText(img, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return img

def load_global_model():
    """Load global face recognition model"""
    global global_face_recognizer, user_labels, label_names
    
    # Initialize recognizer
    global_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load all users and their face data
    faces_data = []
    labels = []
    current_label = 0
    
    for username in os.listdir(USER_DIR):
        user_folder = os.path.join(USER_DIR, username)
        if os.path.isdir(user_folder):
            user_labels[username] = current_label
            label_names[current_label] = username
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            for img_file in os.listdir(user_folder):
                if img_file.endswith('.jpg'):
                    path = os.path.join(user_folder, img_file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)
                        
                        for (x, y, w, h) in detected_faces:
                            face_roi = img[y:y+h, x:x+w]
                            face_roi = cv2.resize(face_roi, (100, 100))
                            faces_data.append(face_roi)
                            labels.append(current_label)
            
            current_label += 1
    
    if faces_data:
        global_face_recognizer.train(faces_data, np.array(labels))
        global_face_recognizer.save("global_face_model.yml")
        
        with open("label_mappings.pkl", "wb") as f:
            pickle.dump({"user_labels": user_labels, "label_names": label_names}, f)

def load_existing_global_model():
    """Load existing global model if available"""
    global global_face_recognizer, user_labels, label_names
    
    if os.path.exists("global_face_model.yml") and os.path.exists("label_mappings.pkl"):
        global_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        global_face_recognizer.read("global_face_model.yml")
        
        with open("label_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
            user_labels = mappings["user_labels"]
            label_names = mappings["label_names"]
        return True
    return False

@app.route('/progress')
def get_progress():
    return jsonify(capture_progress)

@app.route('/')
def index():
    if not load_existing_global_model():
        load_global_model()
    return render_template('index.html')

@app.route('/form/<mode>', methods=['GET', 'POST'])
def form_page(mode):
    if request.method == 'POST':
        username = request.form['username'].strip()

        if mode == "signup":
            user_folder = os.path.join(USER_DIR, username)
            if os.path.exists(user_folder):
                return render_template('result.html', status="fail",
                                       message=f"User '{username}' already exists. Please Sign In.")
            os.makedirs(user_folder)
            return redirect(url_for('camera', mode="capture", username=username))

        elif mode == "signin":
            user_folder = os.path.join(USER_DIR, username)
            if not os.path.exists(user_folder):
                return render_template('result.html', status="fail",
                                       message=f"User '{username}' not found. Please sign up first.")
            return redirect(url_for('camera', mode="verify", username=username))

    registered_users = []
    if mode == "signin" and os.path.exists(USER_DIR):
        registered_users = [d for d in os.listdir(USER_DIR) 
                          if os.path.isdir(os.path.join(USER_DIR, d))]
    
    return render_template('form.html', mode=mode, registered_users=registered_users)

@app.route('/camera/<mode>/<username>')
def camera(mode, username):
    return render_template('camera.html', mode=mode, username=username)

def gen_mock_capture(username):
    """Mock capture function for demo"""
    global capture_progress
    frames = 0
    
    capture_progress = {'current': 0, 'total': 300, 'status': 'capturing'}
    user_folder = os.path.join(USER_DIR, username)
    
    while frames < 300:
        frames += 1
        capture_progress['current'] = frames
        
        # Create mock face image
        frame = create_mock_face_image()
        
        # Add progress text
        progress_text = f"MOCK: Capturing {frames}/300 images"
        cv2.putText(frame, progress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save mock image every 10 frames
        if frames % 10 == 0:
            mock_face = create_mock_face_image()
            cv2.imwrite(os.path.join(user_folder, f"{frames//10}.jpg"), mock_face)
        
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        
        # Simulate real-time capture delay
        import time
        time.sleep(0.1)
    
    capture_progress['status'] = 'training'
    
    # Create mock trained model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces_data = []
    labels = []
    
    for i in range(30):  # Use the saved mock images
        mock_img = create_mock_face_image()
        gray = cv2.cvtColor(mock_img, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(gray[140:340, 220:420], (100, 100))
        faces_data.append(face_roi)
        labels.append(0)
    
    if faces_data:
        face_recognizer.train(faces_data, np.array(labels))
        face_recognizer.save(os.path.join(user_folder, "face_model.yml"))
    
    load_global_model()
    capture_progress['status'] = 'completed'

def gen_mock_verify(username=None):
    """Mock verification function"""
    global global_face_recognizer, label_names
    
    while True:
        frame = create_mock_face_image()
        
        # Simulate face detection
        cv2.rectangle(frame, (220, 140), (420, 340), (0, 255, 0), 2)
        
        # Add mock recognition
        if username and username in user_labels:
            name = username
            color = (0, 255, 0)
        else:
            name = "Demo User"
            color = (0, 255, 0)
        
        cv2.putText(frame, f"{name} (MOCK)", (225, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "DEMO MODE - No Real Camera", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        
        import time
        time.sleep(0.1)

@app.route('/video/<mode>/<username>')
def video_feed(mode, username):
    if mode == "capture":
        return Response(gen_mock_capture(username),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif mode == "verify":
        return Response(gen_mock_verify(username),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/recognize')
def recognize_feed():
    return Response(gen_mock_verify(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance_page():
    return render_template("attendance.html")

@app.route('/mark_attendance')
def mark_attendance():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance_record = {
        'timestamp': timestamp,
        'status': 'Present (MOCK)'
    }
    
    filename = f"attendance_{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}.xlsx"
    filepath = os.path.join(ATTENDANCE_DIR, filename)
    
    df = pd.DataFrame([attendance_record])
    df.to_excel(filepath, index=False)
    
    return jsonify({"status": "success", "message": "Mock attendance marked successfully"})

@app.route('/process/<mode>/<username>')
def process(mode, username):
    if mode == "capture":
        return render_template("result.html", status="success",
                               message=f"✅ Mock registration completed for {username}. Please sign in now.")
    elif mode == "verify":
        return redirect(url_for('home_page', username=username))

@app.route('/home/<username>')
def home_page(username):
    return render_template("home.html", username=username)

@app.route('/result')
def result():
    status = request.args.get("status", "fail")
    message = request.args.get("message", "Something went wrong.")
    return render_template("result.html", status=status, message=message)

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
