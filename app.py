from flask import Flask, render_template, redirect, request
import joblib
import numpy as np
import sklearn

app = Flask(__name__)

#loading the ML model locally
model = joblib.load('kaggle_house_model.pk1')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #extracting the form data
        gr_live_area = float(request.form['GrLivArea'])
        overall_qual = float(request.form['OverallQual'])
        year_built = float(request.form['YearBuilt'])

        #combine this into a numpy input array
        input_data = np.array([[gr_live_area, overall_qual, year_built]])

        #prediction time
        preds = model.predict(input_data)
        return render_template('index.html', prediction_text = f'Predicted Price: ${preds[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)