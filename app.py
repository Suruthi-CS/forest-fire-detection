from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("FFD.keras")  # Ensure the path is correct

# Define image size
img_width, img_height = 150, 150

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            img_path = os.path.join("static/uploads", img_file.filename)
            img_file.save(img_path)

            # Preprocess image
            img = image.load_img(img_path, target_size=(img_width, img_height))
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)[0][0]
            # Adjust the prediction logic here
            if pred > 0.5:  # Adjust the threshold if needed
                prediction = "NO FIRE"
            else:
                prediction = "FIRE"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
