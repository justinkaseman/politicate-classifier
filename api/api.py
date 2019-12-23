"""
api.py
~~~~~~
This module defines a simple REST API for a Machine Learning (ML) model.
"""
from flask import Flask, jsonify, make_response, request
from mlflow import pyfunc
import os
from pandas import DataFrame

app = Flask(__name__)

model = mlflow.sklearn.load_model(
    ""
)


@app.route('/classify', methods=['POST'])
# Use input validation middleware here
def classify():
    try:
        features = DataFrame(request.json)
        classification = model.predict(features).tolist()
        return make_response(jsonify({'classification': 'Test response'}))
    except ValueError:
        raise RuntimeError('Features are not in the correct format.')


@app.route('/ready')
def readiness_check():
    if model.is_ready():
        return Response("", status=200)
    else:
        return Response("", status=503)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
