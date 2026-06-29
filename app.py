from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from predict import predict_skin_tone
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template("index.html")   # ✅ FIXED


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_skin_tone(filepath)

    return render_template(
        "index.html",   # ✅ FIXED
        prediction=result,
        image=filename
    )


if __name__ == "__main__":
    app.run(debug=True)