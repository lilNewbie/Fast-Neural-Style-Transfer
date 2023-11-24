from flask import Flask, render_template, redirect, request, url_for, flash
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
if ~os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.secret_key = "secretkey"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fast_style_transfer(content, style):
    
    content_path = 'static/uploads/' + content
    style_path = 'static/uploads/' + style

    content_img = cv2.cvtColor(cv2.imread(content_path),cv2.COLOR_BGR2RGB)
    style_img = cv2.cvtColor(cv2.imread(style_path),cv2.COLOR_BGR2RGB)

    style_img = cv2.resize(style_img,dsize=content_img.shape[:-1])

    content_img = content_img.astype('float32')[np.newaxis,...]/255.0
    style_img = style_img.astype('float32')[np.newaxis,...]/255.0
    
    generated_img = hub_module(tf.constant(content_img), tf.constant(style_img))[0]
    generated_img = tf.keras.preprocessing.image.array_to_img(generated_img[0])

    generated = 'static/uploads/Generated.png'
    tf.keras.utils.save_img(generated, generated_img)
    
    return content, style, 'Generated.png'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_img():
    if 'file1' not in request.files:
        flash('No Content image available')
        return redirect(request.url)
    elif 'file2' not in request.files:
        flash('No Style image available')
        return redirect(request.url)

    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename=='' :
        flash('No Content image selected for uploading')
        return redirect(request.url)

    if file2.filename=='' :
        flash('No Style image selected for uploading')
        return redirect(request.url)

    if (file1 and allowed_file(file1.filename)) and (file2 and allowed_file(file2.filename)):
        filename1 = secure_filename('ContentImage.'+file1.filename.rsplit('.')[1])
        filename2 = secure_filename('StyleImage.'+file2.filename.rsplit('.')[1])
        
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        print(app.config['UPLOAD_FOLDER'])
        filenameList =  fast_style_transfer(filename1, filename2)
        return render_template('result.html', filename=filenameList)
    else:
        flash('Allowed image types are .pmg, ,jpeg, .jpg')
        return redirect(request.url)
    
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)



if __name__=="__main__":
    app.run(debug=True)