from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import random

app = Flask(__name__)

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = [
    "apple", "apricot", "avocado", "banana", "blue berry", "clementine",
    "corn", "kiwi", "lemon", "mango", "orange", "onion", "peach", "pear", "watermelon"
]

# Function to preprocess the image
def preprocess_image(image_path):
    size = (224, 224)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    class_name = None
    warning_message = None

    if request.method == "POST":
        # Handle image upload
        if "file" not in request.files or request.files["file"].filename == "":
            warning_message = "Please select a file before submitting!"
        else:
            file = request.files["file"]
            file_path = os.path.join("static", "temp.jpg")  # Save uploaded file temporarily
            file.save(file_path)

            # Preprocess the image and make prediction
            data = preprocess_image(file_path)
            prediction = model.predict(data)

            # Get the predicted class
            index = np.argmax(prediction)
            class_name = class_names[index]
            image_path = f"/static/temp.jpg"

    return render_template(
        "index.html", 
        image_path=image_path, 
        class_name=class_name, 
        warning_message=warning_message
    )

# Route for random test image prediction
@app.route("/random", methods=["GET"])
def random_test():
    test_folder = os.path.join("static", "test")
    if not os.path.exists(test_folder):
        return "Test folder not found", 400

    # Choose a random image from the test folder
    test_images = [f for f in os.listdir(test_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not test_images:
        return "No test images found", 400

    random_image = random.choice(test_images)
    file_path = os.path.join(test_folder, random_image)

    # Preprocess the image and make prediction
    data = preprocess_image(file_path)
    prediction = model.predict(data)

    # Get the predicted class
    index = np.argmax(prediction)
    class_name = class_names[index]
    image_path = f"/static/test/{random_image}"

    return render_template("index.html", image_path=image_path, class_name=class_name)

if __name__ == "__main__":
    # Ensure the static/test directory exists
    os.makedirs(os.path.join("static", "test"), exist_ok=True)
    app.run(debug=True)
