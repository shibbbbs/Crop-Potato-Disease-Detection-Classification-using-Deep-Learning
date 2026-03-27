from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# CONFIG
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]
                                                                                                                                                                                                         
IMAGE_SIZE = 256   # should be 256 not 255

# ✅ Load model safely (fix for new keras error)
model = tf.keras.models.load_model(
    "model.h5",
    compile=False
)


# ---------- Prediction function ----------
def predict(img_path):

    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence


# ---------- Home route ----------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html", message="No file")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", message="No file selected")

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            predicted_class, confidence = predict(filepath)

            return render_template(
                "index.html",
                image_path=filepath,
                predicted_label=predicted_class,
                confidence=confidence
            )

    return render_template("index.html")


# ---------- Allowed file ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in [
        "png",
        "jpg",
        "jpeg","jfif"
    ]


# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)