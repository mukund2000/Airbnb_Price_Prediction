# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:20:27 2020

@author: Mukund Rastogi
"""

import numpy as np
import math
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_pred.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    output=math.exp(output)
    return render_template('index.html', prediction_text='Air BnB House prediction price $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)