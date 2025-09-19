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

video_capture = None
capture_progress = {'current': 0, 'total': 300, 'status': 'idle'}
attendance_data = []
global_face_recognizer = None
user_labels = {}
label_names = {}


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
        # Save global model
        global_face_recognizer.save("global_face_model.yml")
        
        # Save label mappings
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

# -------------------- PROGRESS API --------------------
@app.route('/progress')
def get_progress():
    return jsonify(capture_progress)

# -------------------- LANDING --------------------
@app.route('/')
def index():
    # Load existing global model on startup
    if not load_existing_global_model():
        load_global_model()
    return render_template('index.html')


# -------------------- FORM PAGE --------------------
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

    # Get list of registered users for signin dropdown
    registered_users = []
    if mode == "signin" and os.path.exists(USER_DIR):
        registered_users = [d for d in os.listdir(USER_DIR) 
                          if os.path.isdir(os.path.join(USER_DIR, d))]
    
    return render_template('form.html', mode=mode, registered_users=registered_users)


# -------------------- CAMERA --------------------
@app.route('/camera/<mode>/<username>')
def camera(mode, username):
    return render_template('camera.html', mode=mode, username=username)


def gen_capture(username):
    """Collect 300 frames and save images for training"""
    global video_capture, capture_progress
    frames = 0
    
    # Reset progress
    capture_progress = {'current': 0, 'total': 300, 'status': 'capturing'}
    
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    user_folder = os.path.join(USER_DIR, username)
    
    while frames < 300:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw face rectangles and capture images
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Capture image only if faces are detected and we haven't reached the limit
        if len(faces) > 0 and frames < 300:
            frames += 1
            capture_progress['current'] = frames
            
            # Save the entire frame (not just face region for better training)
            cv2.imwrite(os.path.join(user_folder, f"{frames}.jpg"), frame)

        # Add progress text to frame
        progress_text = f"Capturing: {frames}/300 images"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instruction text
        instruction_text = "Please stay still while we capture your face..."
        cv2.putText(frame, instruction_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

        # Stop once we have 300 images
        if frames >= 300:
            break

    # Update progress to training
    capture_progress['status'] = 'training'
    
    # Create and train individual face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces_data = []
    labels = []
    
    for img_file in os.listdir(user_folder):
        if img_file.endswith('.jpg'):
            path = os.path.join(user_folder, img_file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize to standard size
                img_resized = cv2.resize(img, (100, 100))
                faces_data.append(img_resized)
                labels.append(0)  # Single user, so label is 0

    if faces_data:
        face_recognizer.train(faces_data, np.array(labels))
        face_recognizer.save(os.path.join(user_folder, "face_model.yml"))
    
    # Rebuild global model with new user
    load_global_model()
    
    # Mark as completed
    capture_progress['status'] = 'completed'

    # ✅ close camera immediately
    video_capture.release()
    video_capture = None
    cv2.destroyAllWindows()


def gen_verify(username=None):
    """Multi-face recognition with attendance marking"""
    global video_capture, global_face_recognizer, label_names, attendance_data
    
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    confidence_threshold = 80  # Lower means more confident
    
    # Store recognized faces in current session
    session_attendance = set()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            confidence_text = ""
            
            if global_face_recognizer is not None:
                try:
                    label, confidence = global_face_recognizer.predict(face_roi)
                    
                    if confidence < confidence_threshold and label in label_names:
                        name = label_names[label]
                        color = (0, 255, 0)  # Green for recognized
                        confidence_text = f" ({confidence:.1f})"
                        
                        # Add to session attendance
                        session_attendance.add(name)
                    else:
                        confidence_text = f" ({confidence:.1f})"
                except:
                    pass
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw name label with background
            label_text = name + confidence_text
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background rectangle for text
            cv2.rectangle(frame, (x, y-text_height-10), (x+text_width, y), color, -1)
            
            # White text on colored background
            cv2.putText(frame, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add attendance instructions
        instruction_text = "Press 'M' to mark attendance for recognized faces"
        cv2.putText(frame, instruction_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show current session attendance
        if session_attendance:
            attendance_text = f"Recognized: {', '.join(session_attendance)}"
            cv2.putText(frame, attendance_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

# Add attendance marking endpoint
@app.route('/mark_attendance')
def mark_attendance():
    """Mark attendance for currently recognized faces"""
    global attendance_data
    
    # This would be called via AJAX when user presses a button
    # For now, we'll create a simple attendance record
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # In a real implementation, you'd get the current recognized faces
    # For demo, we'll just mark the current user
    attendance_record = {
        'timestamp': timestamp,
        'status': 'Present'
    }
    
    # Save to Excel file
    filename = f"attendance_{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}.xlsx"
    filepath = os.path.join(ATTENDANCE_DIR, filename)
    
    df = pd.DataFrame([attendance_record])
    df.to_excel(filepath, index=False)
    
    return jsonify({"status": "success", "message": "Attendance marked successfully"})

@app.route('/video/<mode>/<username>')
def video_feed(mode, username):
    if mode == "capture":
        return Response(gen_capture(username),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif mode == "verify":
        return Response(gen_verify(username),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/recognize')
def recognize_feed():
    """General face recognition feed for attendance"""
    return Response(gen_verify(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process/<mode>/<username>')
def process(mode, username):
    if mode == "capture":
        return render_template("result.html", status="success",
                               message=f"✅ Registration completed for {username}. Please sign in now.")
    elif mode == "verify":
        return redirect(url_for('home_page', username=username))


# -------------------- ATTENDANCE RECOGNITION --------------------
@app.route('/attendance')
def attendance_page():
    """Page for multi-face recognition and attendance marking"""
    return render_template("attendance.html")

# -------------------- HOME --------------------
@app.route('/home/<username>')
def home_page(username):
    return render_template("home.html", username=username)


# -------------------- RESULT --------------------
@app.route('/result')
def result():
    from flask import request
    status = request.args.get("status", "fail")
    message = request.args.get("message", "Something went wrong.")
    return render_template("result.html", status=status, message=message)


if __name__ == "__main__":
    import os
    # Production configuration
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Always bind to all interfaces for remote access
    
    app.run(host=host, port=port, debug=debug_mode)
