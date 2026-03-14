# AI Gender Detection Web Application

This project is a **Flask-based AI web application** that predicts gender from facial images using a deep learning model.  
The system allows users to upload an image or capture a photo using their camera and receive a prediction along with a confidence score.

The application combines **computer vision and web development** to demonstrate how machine learning models can be integrated into interactive web interfaces.

---

## Features

- Image Upload Gender Detection
- Live Camera Capture & Prediction
- Face Detection before classification
- Confidence score for predictions
- Prediction History tracking
- Multi-page web interface
- Brown–Caramel themed UI design

---

## Tech Stack

- **Python**
- **Flask**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **HTML**
- **CSS**
- **Bootstrap**
- **JavaScript**

---

## Project Structure

```
gender_classification/
│
├── app.py
│
├── model/
│   └── gender_model.h5
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── camera.js
│   └── uploads/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── camera.html
│   ├── history.html
│   ├── result.html
│   └── about.html
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/manjusrid0/gender_classification.git
cd gender_classification
```

Install required dependencies:

```bash
pip install flask tensorflow numpy pillow opencv-python
```

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## How It Works

1. The user uploads an image or captures a photo from the camera.
2. The system detects the face using OpenCV.
3. The detected face is resized and normalized.
4. The pre-trained deep learning model predicts gender.
5. The result and confidence score are displayed.

---

## Example Workflow

```
Upload Image / Capture Camera
          ↓
Face Detection
          ↓
Image Preprocessing
          ↓
Deep Learning Model Prediction
          ↓
Result + Confidence Score
```

---

## Future Improvements

- Real-time camera prediction
- Age detection
- Emotion detection
- Improved model accuracy
- Cloud deployment

---

## Author

**Manjusrid**

GitHub:  
https://github.com/manjusrid0

---

## License

This project is for **educational and research purposes**.
