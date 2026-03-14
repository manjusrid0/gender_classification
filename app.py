from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import base64
import io
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model/gender_model.h5")

IMG_SIZE = 128

history = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(image):

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return image

    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]

    return Image.fromarray(face)


def prepare_image(image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict_gender(image):

    image = detect_face(image)

    processed = prepare_image(image)

    prediction = model.predict(processed)[0][0]

    gender = "Male" if prediction > 0.5 else "Female"

    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return gender, round(confidence * 100, 2)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/camera")
def camera_page():
    return render_template("camera.html")

@app.route("/history")
def history_page():
    return render_template("history.html", history=history)

@app.route("/about")
def about_page():
    return render_template("about.html")
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")

    if not file or file.filename == "":
        return "No image uploaded"

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    image = Image.open(path)

    gender, confidence = predict_gender(image)

    history.append({
        "gender": gender,
        "confidence": confidence,
        "image": path
    })

    return render_template(
        "result.html",
        prediction=gender,
        confidence=confidence,
        image_path=path,
        history=history
    )


@app.route("/camera_predict", methods=["POST"])
def camera_predict():

    data = request.get_json()

    image_data = data["image"]

    encoded_data = image_data.split(",")[1]

    decoded = base64.b64decode(encoded_data)

    image = Image.open(io.BytesIO(decoded))

    gender, confidence = predict_gender(image)

    history.append({
        "gender": gender,
        "confidence": confidence,
        "image": ""
    })

    return jsonify({
        "prediction": gender,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)