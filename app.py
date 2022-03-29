from flask import Flask, render_template, request
import numpy as np
import math
import pickle
from collections.abc import Mapping

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    op = round(prediction[0], 2)

    return render_template('index.html', prediction_text = "Number of Weekly Riders is {}".format(math.floor(op)))

if __name__ == '__main__':
    app.run(debug=True)

