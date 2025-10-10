"""
OPTIMIZED CPU FACE RECOGNITION SYSTEM
Fixed Issues:
- Proper FAISS cosine similarity
- Faster capture process
- Better recognition accuracy
- Optimized CPU performance
"""

import os, cv2, json, time, base64, secrets, hashlib, sqlite3, threading, warnings
import numpy as np
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

# Deep learning (CPU-optimized)
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss

# Optional: MediaPipe for face mesh
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] MediaPipe not available - eye detection simplified")

# Check if InsightFace is available (though this app uses FaceNet)
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except:
    INSIGHTFACE_AVAILABLE = False

from skimage.feature import local_binary_pattern

# ===== App Configuration =====
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

USER_DIR = "users"
ATTENDANCE_DIR = "attendance_collections"
DATABASE_FILE = "face_recognition.db"

# Optimized Thresholds
RECOGNITION_THRESHOLD = 0.70  # Lower for better matching (cosine similarity)
ATTENDANCE_THRESHOLD = 0.60     # Confidence threshold for attendance
HIGH_QUALITY_THRESHOLD = 0.50   # Quality gate threshold
DETECTION_CONFIDENCE = 0.90     # Face detection confidence
LIVENESS_THRESHOLD = 0.55       # Anti-spoofing threshold
EYES_OPEN_THRESHOLD = 0.40      # Eye openness threshold (relaxed)

# Performance settings
CAPTURE_IMAGES = 5              # Reduced from 12 for faster enrollment
LIVENESS_FRAME_BUFFER = 3       # Reduced buffer size
TEMPORAL_WINDOW = 5             # Smaller temporal window

os.makedirs(USER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Force CPU mode
device = torch.device('cpu')
torch.set_num_threads(4)  # Optimize CPU threads

# Global models
mtcnn = None
facenet = None
faiss_index = None
user_id_to_name = {}
face_mesh_api = None

capture_progress = {'current': 0, 'total': CAPTURE_IMAGES, 'status': 'idle', 'message': ''}
attendance_session = {'users': set(), 'start_time': None}

print("=" * 80)
print("OPTIMIZED FACE RECOGNITION SYSTEM")
print(f"Device: CPU (optimized)")
print(f"Recognition Threshold: {RECOGNITION_THRESHOLD}")
print(f"Capture Images: {CAPTURE_IMAGES}")
print("=" * 80)

# ========================
# Utilities
# ========================

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

class ImageEnhancer:
    @staticmethod
    def quick_enhance(frame):
        """Faster enhancement for real-time processing"""
        try:
            # Simple CLAHE for contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        except:
            return frame

# ========================
# Simplified Anti-Spoofing
# ========================
class SimpleLivenessDetector:
    def __init__(self):
        self.frame_buffer = []
    
    def check_liveness(self, frame):
        """Simplified liveness check for speed"""
        try:
            # Add frame to buffer
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            self.frame_buffer.append(gray)
            if len(self.frame_buffer) > LIVENESS_FRAME_BUFFER:
                self.frame_buffer.pop(0)
            
            if len(self.frame_buffer) < 2:
                return 0.6, "Initializing"
            
            # Check motion between frames
            motion = np.mean(cv2.absdiff(self.frame_buffer[-1], self.frame_buffer[-2]))
            
            # Simple texture analysis
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            texture_var = np.var(lbp)
            
            # Combined score
            motion_score = min(motion / 10.0, 1.0)
            texture_score = min(texture_var / 50.0, 1.0)
            
            score = (motion_score * 0.5 + texture_score * 0.5)
            status = "Live" if score >= LIVENESS_THRESHOLD else "Check lighting"
            
            return float(score), status
        except:
            return 0.6, "Error"

# ========================
# Face Mesh for Eye Detection
# ========================
class EyeDetector:
    def __init__(self):
        self.mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_face_mesh = mp.solutions.face_mesh
                self.mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,  # Faster without refinement
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except:
                self.mesh = None
    
    def eyes_open_score(self, frame):
        """Simplified eye detection"""
        if self.mesh is None:
            return 0.7  # Default to open if no detector
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                return 0.7
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Simple eye aspect ratio
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
            
            def eye_aspect_ratio(eye):
                vertical = abs(eye[1].y - eye[5].y) + abs(eye[2].y - eye[4].y)
                horizontal = abs(eye[0].x - eye[3].x) * 2
                return vertical / (horizontal + 0.001)
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Map EAR to score (0.15 = closed, 0.30 = open)
            score = np.clip((avg_ear - 0.15) / 0.15, 0, 1)
            return float(score)
        except:
            return 0.7

# ========================
# Core Face Recognition System
# ========================
class FaceRecognitionSystem:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.liveness = SimpleLivenessDetector()
        self.eye_detector = EyeDetector()
        self.initialize_models()
        self.initialize_database()
        self.load_embeddings()
    
    def initialize_models(self):
        """Initialize models with CPU optimization"""
        global mtcnn, facenet
        
        # MTCNN for face detection
        mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,  # Larger min size for speed
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=False,  # Single face for speed
            device=device
        )
        
        # FaceNet for embeddings
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        print("✅ Models loaded (CPU-optimized)")
    
    def initialize_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP,
            embedding_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1
        )""")
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            embedding BLOB NOT NULL,
            quality_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            session_id TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS admins(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Create default admin
        c.execute("SELECT COUNT(*) FROM admins")
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                     ('admin', hash_password('admin123')))
        
        conn.commit()
        conn.close()
    
    def load_embeddings(self):
        """Load embeddings and build FAISS index"""
        global faiss_index, user_id_to_name
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        c.execute("""
        SELECT u.id, u.username, fe.embedding
        FROM users u 
        JOIN face_embeddings fe ON u.id = fe.user_id
        WHERE u.is_active = 1
        ORDER BY u.id, fe.quality_score DESC
        """)
        
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            faiss_index = None
            user_id_to_name = {}
            print("⚠️ No embeddings found")
            return False
        
        # Group by user and take best embeddings
        user_embeddings = defaultdict(list)
        for uid, username, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            user_embeddings[uid].append((username, emb))
        
        # Build index
        all_embeddings = []
        user_id_to_name = {}
        idx = 0
        
        for uid, emb_list in user_embeddings.items():
            username = emb_list[0][0]
            # Use up to 3 embeddings per user
            for _, emb in emb_list[:3]:
                all_embeddings.append(emb)
                user_id_to_name[idx] = username
                idx += 1
        
        if all_embeddings:
            X = np.array(all_embeddings).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(X)
            
            # Use L2 distance on normalized vectors (equivalent to cosine)
            d = X.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(X)
            
            faiss_index = index
            print(f"✅ FAISS index: {len(all_embeddings)} embeddings, {len(user_embeddings)} users")
            return True
        
        return False
    
    def detect_faces(self, image):
        """Fast face detection"""
        try:
            # Convert to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            # Detect faces
            boxes, probs = mtcnn.detect(pil_image)
            
            if boxes is None:
                return []
            
            faces = []
            for box, prob in zip(boxes, probs):
                if prob > DETECTION_CONFIDENCE:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    w, h = x2 - x1, y2 - y1
                    faces.append({
                        'box': [x1, y1, w, h],
                        'confidence': float(prob)
                    })
            
            return faces
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def calculate_quality(self, image, face_box):
        """Simplified quality calculation"""
        try:
            x, y, w, h = face_box
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                return 0.0
            
            # Check sharpness
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian / 100, 1.0)
            
            # Check brightness
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Check size
            size_ratio = (w * h) / (image.shape[0] * image.shape[1])
            size_score = min(size_ratio * 20, 1.0)
            
            return sharpness * 0.4 + brightness_score * 0.3 + size_score * 0.3
        except:
            return 0.0
    
    def extract_embedding(self, image, face_box=None):
        """Extract face embedding"""
        try:
            # Crop face region
            if face_box:
                x, y, w, h = face_box
                margin = int(0.2 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                face_img = image[y1:y2, x1:x2]
            else:
                face_img = image
            
            # Convert to RGB
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            # Get face tensor
            face_tensor = mtcnn(pil_image)
            
            if face_tensor is None:
                return None
            
            # Ensure correct shape
            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = facenet(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def save_embedding(self, username, embedding, quality_score):
        """Save embedding to database"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            c = conn.cursor()
            
            # Get or create user
            c.execute('SELECT id FROM users WHERE username = ?', (username,))
            row = c.fetchone()
            
            if row:
                user_id = row[0]
            else:
                c.execute('INSERT INTO users (username) VALUES (?)', (username,))
                user_id = c.lastrowid
            
            # Save embedding
            blob = embedding.tobytes()
            c.execute("""
            INSERT INTO face_embeddings (user_id, embedding, quality_score)
            VALUES (?, ?, ?)
            """, (user_id, blob, quality_score))
            
            # Update embedding count
            c.execute("""
            UPDATE users SET embedding_count = (
                SELECT COUNT(*) FROM face_embeddings WHERE user_id = ?
            ) WHERE id = ?
            """, (user_id, user_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def recognize_face(self, embedding, threshold=RECOGNITION_THRESHOLD):
        """Recognize face using FAISS"""
        try:
            if faiss_index is None or embedding is None:
                return "Unknown", 0.0
            
            # Prepare query
            query = embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query)
            
            # Search
            k = min(3, faiss_index.ntotal)
            distances, indices = faiss_index.search(query, k)
            
            if len(indices[0]) == 0:
                return "Unknown", 0.0
            
            # Convert L2 distance to cosine similarity
            # For normalized vectors: cosine_sim = 1 - (L2_distance^2 / 2)
            distance = distances[0][0]
            similarity = 1 - (distance ** 2) / 2
            
            if similarity >= threshold:
                idx = indices[0][0]
                username = user_id_to_name.get(idx, "Unknown")
                confidence = similarity * 100
                return username, confidence
            
            return "Unknown", 0.0
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0.0

# Initialize system
face_system = FaceRecognitionSystem()

# ========================
# Routes
# ========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form/<mode>', methods=['GET', 'POST'])
def form_page(mode):
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        
        if not username:
            return render_template('result.html', 
                                 status="fail", 
                                 message="Username is required")
        
        if mode == 'signup':
            # Check if user exists
            conn = sqlite3.connect(DATABASE_FILE)
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE username = ?', (username,))
            exists = c.fetchone()
            conn.close()
            
            if exists:
                return render_template('result.html',
                                     status="fail",
                                     message=f"User '{username}' already exists")
            
            # Reset progress
            global capture_progress
            capture_progress = {
                'current': 0,
                'total': CAPTURE_IMAGES,
                'status': 'idle',
                'message': ''
            }
            
            # Create user directory
            os.makedirs(os.path.join(USER_DIR, username), exist_ok=True)
            
            return redirect(url_for('camera', mode='capture', username=username))
        
        elif mode == 'signin':
            # Check if user exists
            conn = sqlite3.connect(DATABASE_FILE)
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE username = ? AND is_active = 1', 
                     (username,))
            exists = c.fetchone()
            conn.close()
            
            if not exists:
                return render_template('result.html',
                                     status="fail",
                                     message=f"User '{username}' not found")
            
            return redirect(url_for('camera', mode='verify', username=username))
    
    # GET request
    registered = []
    if mode == 'signin':
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute('SELECT username FROM users WHERE is_active = 1 ORDER BY username')
        registered = [r[0] for r in c.fetchall()]
        conn.close()
    
    return render_template('form.html', mode=mode, registered_users=registered)

@app.route('/camera/<mode>/<username>')
def camera(mode, username):
    return render_template('camera.html', mode=mode, username=username)

@app.route('/progress')
def get_progress():
    return jsonify(capture_progress)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        data = request.json
        frame_b64 = data.get('frame')
        mode = data.get('mode')
        username = data.get('username', '')
        
        if not frame_b64:
            return jsonify({"error": "No frame data"}), 400
        
        # Decode frame
        img_bytes = base64.b64decode(frame_b64.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if mode == 'capture':
            return process_capture_frame(frame, username)
        else:
            return process_recognition_frame(frame)
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

def process_capture_frame(frame, username):
    """Process frame for enrollment"""
    global capture_progress
    
    try:
        # Check if already done
        if capture_progress['current'] >= capture_progress['total']:
            return jsonify({
                "status": "completed",
                "message": "Registration completed"
            })
        
        # Detect faces
        faces = face_system.detect_faces(frame)
        if not faces:
            return jsonify({
                "status": "no_face",
                "message": "No face detected"
            })
        
        face = faces[0]
        face_box = face['box']
        
        # Check quality
        quality = face_system.calculate_quality(frame, face_box)
        if quality < HIGH_QUALITY_THRESHOLD:
            return jsonify({
                "status": "low_quality",
                "message": f"Quality: {int(quality*100)}%. Improve lighting",
                "quality_score": round(quality, 2)
            })
        
        # Check eyes (relaxed for enrollment)
        eye_score = face_system.eye_detector.eyes_open_score(frame)
        if eye_score < EYES_OPEN_THRESHOLD:
            return jsonify({
                "status": "eyes_closed",
                "message": "Please keep eyes open",
                "eye_score": round(eye_score, 2)
            })
        
        # Extract embedding
        embedding = face_system.extract_embedding(frame, face_box)
        if embedding is None:
            return jsonify({
                "status": "embedding_failed",
                "message": "Could not extract features"
            })
        
        # Save embedding
        success = face_system.save_embedding(username, embedding, quality)
        if not success:
            return jsonify({
                "status": "save_failed",
                "message": "Could not save data"
            })
        
        # Update progress
        capture_progress['current'] += 1
        capture_progress['status'] = 'capturing'
        capture_progress['message'] = f"Captured {capture_progress['current']}/{capture_progress['total']}"
        
        # Save image
        img_path = os.path.join(USER_DIR, username, f"{capture_progress['current']}.jpg")
        cv2.imwrite(img_path, frame)
        
        # Check if done
        if capture_progress['current'] >= capture_progress['total']:
            capture_progress['status'] = 'training'
            capture_progress['message'] = 'Finalizing registration...'
            
            # Rebuild index in background
            threading.Thread(target=rebuild_index_async, daemon=True).start()
        
        return jsonify({
            "status": "success",
            "faces_detected": 1,
            "images_captured": capture_progress['current'],
            "quality_score": round(quality, 2),
            "eye_score": round(eye_score, 2),
            "progress": capture_progress,
            "face_positions": [{
                "x": face_box[0],
                "y": face_box[1], 
                "w": face_box[2],
                "h": face_box[3]
            }]
        })
    
    except Exception as e:
        print(f"Capture error: {e}")
        return jsonify({"status": "error", "message": str(e)})

def process_recognition_frame(frame):
    """Process frame for recognition"""
    try:
        # Detect faces
        faces = face_system.detect_faces(frame)
        if not faces:
            return jsonify({
                "status": "no_face",
                "faces": [],
                "total_faces_detected": 0
            })
        
        results = []
        for face in faces:
            face_box = face['box']
            
            # Check eyes
            eye_score = face_system.eye_detector.eyes_open_score(frame)
            
            # Extract embedding
            embedding = face_system.extract_embedding(frame, face_box)
            if embedding is None:
                results.append({
                    "name": "Unknown",
                    "confidence": 0.0,
                    "position": {
                        "x": face_box[0],
                        "y": face_box[1],
                        "w": face_box[2],
                        "h": face_box[3]
                    }
                })
                continue
            
            # Recognize
            name, confidence = face_system.recognize_face(embedding)
            
            # Check liveness if recognized
            liveness_score = 0.7
            if name != "Unknown":
                liveness_score, _ = face_system.liveness.check_liveness(frame)
            
            results.append({
                "name": name,
                "confidence": round(confidence, 2),
                "position": {
                    "x": face_box[0],
                    "y": face_box[1],
                    "w": face_box[2],
                    "h": face_box[3]
                },
                "eye_score": round(eye_score, 2),
                "liveness_score": round(liveness_score, 2)
            })
        
        return jsonify({
            "status": "success",
            "faces": results,
            "total_faces_detected": len(faces)
        })
    
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({"status": "error", "message": str(e)})

def rebuild_index_async():
    """Rebuild FAISS index in background"""
    global capture_progress
    try:
        time.sleep(0.5)  # Brief pause
        face_system.load_embeddings()
        capture_progress['status'] = 'completed'
        capture_progress['message'] = 'Registration successful!'
    except Exception as e:
        capture_progress['status'] = 'error'
        capture_progress['message'] = f'Error: {e}'

@app.route('/attendance')
def attendance_page():
    global attendance_session
    attendance_session = {'users': set(), 'start_time': datetime.now()}
    return render_template('attendance.html')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.json
        faces = data.get('faces', [])
        
        marked = []
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        for face in faces:
            name = face.get('name')
            confidence = face.get('confidence', 0)
            
            if name != "Unknown" and confidence >= ATTENDANCE_THRESHOLD * 100:
                if name not in attendance_session['users']:
                    # Get user ID
                    c.execute('SELECT id FROM users WHERE username = ?', (name,))
                    row = c.fetchone()
                    
                    if row:
                        user_id = row[0]
                        c.execute("""
                        INSERT INTO attendance (user_id, username, confidence, session_id)
                        VALUES (?, ?, ?, ?)
                        """, (user_id, name, confidence, session_id))
                        
                        attendance_session['users'].add(name)
                        marked.append({
                            'username': name,
                            'confidence': confidence
                        })
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "marked_users": marked,
            "message": f"Marked attendance for {len(marked)} users"
        })
    
    except Exception as e:
        print(f"Attendance error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ========================
# Admin Routes
# ========================
@app.route('/admin')
def admin_panel():
    return render_template('admin.html')

@app.route('/admin/login', methods=['POST'])
def admin_login():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute('SELECT id, username FROM admins WHERE username = ? AND password_hash = ?',
                 (username, hash_password(password)))
        row = c.fetchone()
        conn.close()
        
        if row:
            session['logged_in'] = True
            session['admin_id'] = row[0]
            session['admin_username'] = row[1]
            session['csrf_token'] = secrets.token_hex(32)
            
            return jsonify({
                "status": "success",
                "message": "Login successful",
                "csrf_token": session['csrf_token']
            })
        else:
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out successfully"})

@app.route('/api/admin/users')
def get_users():
    if not session.get('logged_in'):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("""
        SELECT id, username, full_name, email, created_at, last_seen, 
               embedding_count, is_active
        FROM users 
        ORDER BY created_at DESC
        """)
        
        users = []
        for row in c.fetchall():
            users.append({
                "id": row[0],
                "username": row[1],
                "full_name": row[2],
                "email": row[3],
                "created_at": row[4],
                "last_seen": row[5],
                "embedding_count": row[6],
                "is_active": bool(row[7])
            })
        
        conn.close()
        return jsonify({"status": "success", "users": users})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/admin/users')
def admin_users_page():
    if not session.get('logged_in'):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("""
        SELECT id, username, full_name, email, created_at, last_seen, 
               embedding_count, is_active
        FROM users 
        ORDER BY created_at DESC
        """)
        
        users = []
        for row in c.fetchall():
            users.append({
                "id": row[0],
                "username": row[1],
                "full_name": row[2] or row[1],
                "email": row[3] or "",
                "created_at": row[4],
                "last_seen": row[5] or "Never",
                "embedding_count": row[6] or 0,
                "is_active": bool(row[7])
            })
        
        conn.close()
        return jsonify({"status": "success", "users": users})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if not session.get('logged_in'):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        # Check if user exists
        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        username = user[0]
        
        # Delete user's face embeddings
        c.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,))
        
        # Delete user's attendance records
        c.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
        
        # Delete the user
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        
        # Try to delete user's image directory
        import shutil
        user_dir = os.path.join(USER_DIR, username)
        if os.path.exists(user_dir):
            try:
                shutil.rmtree(user_dir)
            except Exception as e:
                print(f"Warning: Could not delete user directory {user_dir}: {e}")
        
        # Rebuild FAISS index after user deletion
        try:
            rebuild_faiss_index()
        except Exception as e:
            print(f"Warning: Could not rebuild FAISS index: {e}")
        
        return jsonify({
            "status": "success", 
            "message": f"User '{username}' deleted successfully"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def rebuild_faiss_index():
    """Rebuild FAISS index after user changes"""
    global faiss_index, user_id_to_name, user_embeddings_map
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        # Get all active users and their embeddings
        c.execute("""
        SELECT u.username, fe.embedding, fe.quality_score, fe.is_augmented
        FROM users u 
        JOIN face_embeddings fe ON u.id = fe.user_id
        WHERE u.is_active = 1
        ORDER BY u.username, fe.quality_score DESC
        """)
        
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            faiss_index = None
            user_id_to_name = {}
            user_embeddings_map = defaultdict(list)
            print("⚠️ No embeddings found after deletion")
            return
        
        # Organize embeddings by user
        user_embeddings = defaultdict(list)
        for username, blob, quality, is_aug in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            # Only use high-quality non-augmented embeddings for the index
            if quality > 0.5 and not is_aug:
                user_embeddings[username].append(emb)
        
        # Build FAISS index
        all_embeddings = []
        user_id_to_name = {}
        user_embeddings_map = defaultdict(list)
        idx = 0
        
        for username, embs in user_embeddings.items():
            # Use up to 5 best embeddings per user
            for emb in embs[:5]:
                all_embeddings.append(emb)
                user_id_to_name[idx] = username
                user_embeddings_map[username].append(emb)
                idx += 1
        
        if all_embeddings:
            # Create FAISS index with L2 distance
            X = np.array(all_embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            import faiss
            faiss.normalize_L2(X)
            
            # Use L2 distance on normalized vectors (equivalent to cosine similarity)
            d = X.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(X)
            faiss_index = index
            
            print(f"✅ FAISS index rebuilt: {len(all_embeddings)} embeddings from {len(user_embeddings)} users")
        else:
            faiss_index = None
            print("⚠️ No valid embeddings to rebuild index")
            
    except Exception as e:
        print(f"❌ FAISS rebuild error: {e}")

@app.route('/api/admin/attendance')
def get_attendance():
    if not session.get('logged_in'):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("""
        SELECT a.id, a.username, a.timestamp, a.confidence, a.liveness_score, a.session_id
        FROM attendance a
        ORDER BY a.timestamp DESC
        LIMIT 100
        """)
        
        attendance = []
        for row in c.fetchall():
            attendance.append({
                "id": row[0],
                "username": row[1],
                "timestamp": row[2],
                "confidence": row[3],
                "liveness_score": row[4],
                "session_id": row[5]
            })
        
        conn.close()
        return jsonify({"status": "success", "attendance": attendance})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/admin/system/stats')
def admin_system_stats():
    if not session.get('logged_in'):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        
        # Get user count
        c.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        active_users = c.fetchone()[0]
        
        # Get total user count
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]
        
        # Get total embeddings count
        c.execute("SELECT COUNT(*) FROM face_embeddings")
        total_embeddings = c.fetchone()[0]
        
        # Get attendance count today
        c.execute("SELECT COUNT(*) FROM attendance WHERE DATE(timestamp) = DATE('now')")
        today_attendance = c.fetchone()[0]
        
        # Get total attendance count
        c.execute("SELECT COUNT(*) FROM attendance")
        total_attendance = c.fetchone()[0]
        
        conn.close()
        
        # Format data to match what the admin page expects
        response = {
            "status": "success",
            "stats": {
                "active_users": active_users,
                "total_users": total_users,
                "total_embeddings": total_embeddings,
                "today_attendance": today_attendance,
                "total_attendance": total_attendance,
                "system_health": "healthy",
                "models": {
                    "insightface": INSIGHTFACE_AVAILABLE,
                    "mediapipe": MEDIAPIPE_AVAILABLE,
                    "faiss": faiss_index is not None
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ========================
# Entrypoint
# ========================
if __name__ == "__main__":
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0' if not debug_mode else '127.0.0.1'

    print("\n" + "="*80)
    print("🚀 CPU‑ONLY FACE RECOG with Eyes‑Open Gate")
    print("="*80)
    print(f"📍 http://{host}:{port}")
    print(f"🎯 Cosine threshold: {RECOGNITION_THRESHOLD}")
    print(f"👀 Eyes‑open threshold: {EYES_OPEN_THRESHOLD}")
    print("="*80 + "\n")

    app.run(host=host, port=port, debug=debug_mode, threaded=True)
