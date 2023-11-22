from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_url_path='/static')
model = load_model("BodyFat.h5", compile=False)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = [
            float(request.form['a']),  # Age
            float(request.form['b']),  # Weight
            float(request.form['c']),  # Height
            float(request.form['d']),  # Neck circumference
            float(request.form['e']),  # Chest circumference
            float(request.form['f']),  # Abdomen circumference
            float(request.form['g']),  # Hip circumference
            float(request.form['h']),  # Thigh circumference
            float(request.form['i']),  # Knee circumference
            float(request.form['j']),  # Ankle circumference
            float(request.form['k']),  # Biceps (extended) circumference
            float(request.form['l']),  # Forearm circumference
            float(request.form['m'])  # Wrist circumference
        ]

        prediction = model.predict(tf.expand_dims(input_data, axis=0))
        return render_template('predict.html', prediction=prediction)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
