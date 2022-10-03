import numpy as np
from flask import Flask, request, jsonify,  render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('heart_disease_classifier', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['get','post'])
def predict():
    features = [float(i) for i in request.form.values()]
    col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']
    array_features = pd.DataFrame(features, col)
    prediction = model.predict(array_features.T)
    output = prediction[0]
    if output == 1:
        return render_template('index.html', 
                               prediction_text = 'The patient is not likely to have heart disease!')
    else:
        return render_template('index.html', 
                               prediction_text = 'The patient is likely to have heart disease!')

if __name__=='__main__':
    app.run(debug=True)
