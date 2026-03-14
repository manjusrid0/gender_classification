from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model/gender_model.h5")

img_size = 128

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = load_img(path, target_size=(img_size,img_size))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    gender = "Male" if prediction[0][0] > 0.5 else "Female"

    return render_template("result.html",
                           prediction=gender,
                           image_path=path)

if __name__ == "__main__":
    app.run(debug=True)