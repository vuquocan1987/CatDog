from flask import Flask
from flask import render_template
from flask import request, redirect
from werkzeug.utils import secure_filename
from datetime import datetime
from keras.models import Sequential
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import re

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./IMG/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG","JPG","PNG","GIF"]
app.config["MAX_CONTENT_LENGTH"] = 50*1024*1024
IMAGE_SIZE = 192
model = load_model("catdog_classifier_MobileNetV2.h5")
model.summary()
def allowed_image(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".",1)[1]
    return ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]
def allowed_image_filesize(filesize):
    return int(filesize) <= app.config["MAX_CONTENT_LENGTH"] 
@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0
    return image

@app.route("/upload-image",methods = ["GET","POST"])
def upload_image():
#   test_model()
#    model.summary()
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                print("No filename")
                return redirect(request.url)
            if "filesize" in request.cookies:
                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                path = os.path.join(app.config["IMAGE_UPLOADS"],filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"],filename))
                confidence = classify_img(path)
                
                print("Image saved")
            
            return render_template("public/upload_image.html",confidence = confidence)
    return render_template("public/upload_image.html")
def classify_img(path):
    processed_img = load_and_preprocess_image(path)
    result = model.predict(tf.reshape(processed_img,(1,IMAGE_SIZE,IMAGE_SIZE,3)))
    print(result)
    return result[0]
def test_model():
    model = load_model("catdog_classifier_MobileNetV2.h5")
    model.summary()