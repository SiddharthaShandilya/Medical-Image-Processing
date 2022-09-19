from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
#app = Flask(__name__,template_folder='./flask app')
app = Flask(__name__,)

# Model saved with Keras model.save()
MODEL_PATH = './Artifacts/models/inception_ct.h5'
#MODEL_PATH = './Artifacts/models/xception_chest.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(224, 224))
    image = cv2.imread(img_path) # read file 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255             #normalization
    image = np.expand_dims(image, axis=0)

    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(image, mode='caffe')

    preds = model.predict(image)
    print(f"\n{preds}\n")
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
 
        # Make prediction
        preds = model_predict(file_path, model)
        #result  = preds
        print(f"\n{preds}\n")
        #os.remove(file_path)
        #return str(np.argmax(result))

        probability = preds[0]
        print("Inception Predictions:")
        if probability[0] > 0.5:
            inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
        else:
            inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
        
        return(inception_chest_pred)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        #return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

