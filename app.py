import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('HouseRent_Joblib_XGB.joblib' + '.gz')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    state = request.form['state']
    region = request.form['region']
    print('Region :- ',request.form['region'])
    lat = request.form['lat']
    lat=int(float(lat))
    long = request.form['long']
    long=int(float(long))
    type = request.form['type']
    sqfeet = request.form['sqfeet']
    beds = request.form['beds']
    baths = request.form['baths']
    cats_allowed = request.form['cats_allowed']
    dogs_allowed = request.form['dogs_allowed']
    smoking_allowed = request.form['smoking_allowed']
    wheelchair_access = request.form['wheelchair_access']
    electric_vehicle_charge = request.form['electric_vehicle_charge']
    comes_furnished = request.form['comes_furnished']
    laundry_options = request.form['laundry_options']
    parking_options = request.form['parking_options']

    final_values=[state, region, lat, long, type, sqfeet, beds, baths,
                  cats_allowed, dogs_allowed, smoking_allowed,wheelchair_access,
                  electric_vehicle_charge,comes_furnished,laundry_options,parking_options]

    print(final_values)

    int_features = [int(x) for x in final_values]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='House Rent would be : ${:.2f}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
