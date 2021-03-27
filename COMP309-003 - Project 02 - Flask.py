# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 06:48:46 2020

@author: koola
"""

"--------------------------------------------------"
# 05
"--------------------------------------------------"
##################################################
# a.	Using flask framework arrange to turn your selected machine-learning model into an API.

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            scaled_df = scaler.fit_transform(query)
            # return to data frame
            query = pd.DataFrame(scaled_df, columns=model_columns)
            print(query)
            prediction = list(model.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to bike theft model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    model = joblib.load('C:/COMP309-003 - Project 02/model_prediction_recovery_bicycle.pkl')  
    print ('Model loaded')
    model_columns = joblib.load('C:/COMP309-003 - Project 02/model_prediction_recovery_bicycle_columns.pkl')
    print ('Model columns loaded')
    app.run(port=port, debug=True)