import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from genetic_selection import GeneticSelectionCV
from skimage import feature

app = Flask(__name__)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError('Could not open camera.')
model = None
selected_model = ""
training_data = []
training_labels = []
captured_images = 0
label_mapping = {}
face_recognition_enabled = False

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		return hist

def train_gen_model():
    global model, training_data, training_labels, label_mapping, face_recognition_enabled

    y = np.array(training_labels)

    data = []
    for x in training_data:
        d = x.flatten()
        data.append(d)
    data = np.array(data)
    data = data.astype(float)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    clf_rf = RandomForestClassifier(max_depth=6, random_state=42)

    model = GeneticSelectionCV(
        clf_rf, cv=5, verbose=0,
        scoring="accuracy", max_features=10,
        n_population=50, crossover_proba=0.5,
        mutation_proba=0.2, n_generations=50,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.04,
        tournament_size=5, n_gen_no_change=10,
        caching=True, n_jobs=-1)
    
    model = model.fit(data, y)

    labels = label_encoder.classes_
    label_mapping = {i: label for i, label in enumerate(labels)}
    face_recognition_enabled = True

def train_fp_model():
    global model, training_data, training_labels, label_mapping, face_recognition_enabled

    desc = LocalBinaryPatterns(64, 8)

    X = np.array(training_data)
    y = np.array(training_labels)
    
    features = []
    for x in X:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        f = desc.describe(x)
        if f is not None:
            features.append(f)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, y)

    labels = label_encoder.classes_
    label_mapping = {i: label for i, label in enumerate(labels)}
    face_recognition_enabled = True

def train_knn_model():
    global model, training_data, training_labels, label_mapping, face_recognition_enabled

    X = np.array(training_data)
    y = np.array(training_labels)

    data = []
    for x in training_data:
        d = x.flatten()
        data.append(d)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(data, y)

    labels = label_encoder.classes_
    label_mapping = {i: label for i, label in enumerate(labels)}
    face_recognition_enabled = True

def capture_images(label):
    captured_images = 0

    while captured_images < 200:
        success, frame = camera.read()
        if not success:
            break

        cv2.imshow("Capturing Images...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        resized_frame = cv2.resize(frame, (64, 64))

        training_data.append(resized_frame)
        training_labels.append(label)

        captured_images += 1

    cv2.destroyAllWindows()

def recognize_face(frame):
    global model, selected_model, face_recognition_enabled

    label_encoded = None

    if model is not None:
        #-------------------------------
        # Mô hình haarcascade_frontalface_default.xml là một mô hình được cung cấp sẵn trong thư viện OpenCV. 
        # Nó là một mô hình phân loại cascade được huấn luyện để nhận dạng khuôn mặt trong các hình ảnh.
        # Mô hình cascade là một loại mô hình nhận dạng đặc trưng dựa trên việc sử dụng một tập hợp các phân loại đơn giản (cascade) để tìm kiếm các đặc trưng của đối tượng mong muốn. 
        # Mô hình cascade kết hợp nhiều bộ phân loại đơn giản để xác định các vùng quan tâm trong hình ảnh.
        #--------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (64, 64))

            if selected_model == 'knn':
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
                face_features = face_roi.flatten().reshape(1, -1)
            elif selected_model == 'frequent_pattern':
                desc = LocalBinaryPatterns(64, 8)

                face_features = desc.describe(face_roi)
                face_features = face_features.reshape(1, -1)
            elif selected_model == 'genetic':
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
                face_features = face_roi.flatten().reshape(1, -1)
            
            label_encoded = model.predict(face_features)[0]
            label = label_mapping[label_encoded]

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            raise RuntimeError('Failed to capture frame.')

        if frame is not None:

            frame_with_predictions = frame

            if face_recognition_enabled:
                frame_with_predictions = recognize_face(frame)

            ret, buffer = cv2.imencode('.jpg', frame_with_predictions)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train_model')
def train_model():
    global model, training_data, training_labels, selected_model, face_recognition_enabled

    selected_model = request.args.get('model')
    label = request.args.get('label')

    if len(training_data) < 200:
        return "Insufficient training data. Please capture 200 images first."

    if model is None:
        if selected_model == 'knn':

            train_knn_model()

            return "KNN model trained successfully."
        elif selected_model == 'frequent_pattern':

            train_fp_model()

            return "Frequent Pattern model trained successfully."
        elif selected_model == 'genetic':

            train_gen_model()

            return "Genetic model trained successfully."
        else:
            return "Invalid model selection."
    else:
        return "Model is already trained."

@app.route('/capture_images')
def capture_images_route():
    global captured_images

    label = request.args.get('label')
    if label is None or label.strip() == "":
        return "Invalid label. Please enter a label."

    capture_images(label)

    return "Image capture completed!"

if __name__ == '__main__':
    app.run(debug=True)
