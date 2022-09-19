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

app = Flask(__name__,)

# Model saved with Keras model.save()

# Load your trained model

#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, MODEL_PATH):
    #img = image.load_img(img_path, target_size=(224, 224))
    image = cv2.imread(img_path) # read file 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255             #normalization
    image = np.expand_dims(image, axis=0)

    #loading model
    model = load_model(MODEL_PATH)

    print(f"\n Model loaded from file {MODEL_PATH} \n")

    #predicting the image
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
        models_html = request.form['Model']

        print(" +++ "*10)
        try:
            print(models_html)
            if models_html=='0':
                MODEL_PATH = './Artifacts/models/xception_chest.h5'
            else:
                MODEL_PATH = './Artifacts/models/inception_ct.h5'                
        except:
            print("model is not defined")
        print(" +++ "*10)



        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
 
        # Make prediction
        preds = model_predict(file_path, MODEL_PATH)

        
        print(f"\n{preds}\n")
        #os.remove(file_path)
        #return str(np.argmax(preds))

        probability = preds[0]
        print("Inception Predictions:")
        if probability[0] > 0.5:
            coivd_prediction = str('%.2f' % (probability[0]*100) + '% COVID') 
        else:
            coivd_prediction = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
        
        return(coivd_prediction)

    return None


if __name__ == '__main__':
    app.run(debug=True)

