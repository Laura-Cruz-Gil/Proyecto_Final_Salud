from flask import Flask, redirect, request, render_template, url_for
import joblib
import numpy as np
import os
import pandas as pd
from datetime import date

model_path = os.path.join(os.path.dirname(__file__), '../models/best_random_forest_model.joblib')
model = joblib.load(model_path)

y_train = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/processed/y_train.csv'), delimiter=',')
X_train_with_outliers = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/processed/X_train_with_outliers.csv'), delimiter=',')

TEMPERATURE_MIN = -62.2
TEMPERATURE_MAX =  56.7

app = Flask(__name__)

states_list = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming"
]

states_cols = [f"state_{state}" for state in states_list]
feature_columns = ["epiweek", "temperature_2m_mean", "num_ili", "num_providers"] + states_cols


def validate_request_data(request) -> str:

    """ Validates the request data for the prediction endpoint. """

    message = None
    if not request.form.get('num_ili') or not request.form.get('num_providers'):
        return "Error: All fields are required (num_ili, num_providers)."
    try:
        float(request.form['num_ili'])
        float(request.form['num_providers'])
    except ValueError:
        return "Error: Invalid input. Please enter numeric values for temperature, num_ili, and num_providers."
    
    temperature = request.form['temperature']
    if temperature:
        try:
            temperature = float(temperature)
            if temperature < TEMPERATURE_MIN or temperature > TEMPERATURE_MAX:
                return f"Error: Temperature out of range. Please enter a value between {TEMPERATURE_MIN} and {TEMPERATURE_MAX}."
        except ValueError:
           return "Error: Invalid input. Please enter a numeric value for temperature."

@app.route('/')
def home():
    return redirect(url_for('prediction'))

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Validar los datos solo cuando se envía el formulario.
        error_message = validate_request_data(request)
        if error_message:
            return error_message

        today = date.today()
        year, week, _ = today.isocalendar()
        epiweek_current = int(f"{year}{week:02d}") 

        default_num_ili = X_train_with_outliers['num_ili'].mean()
        default_num_providers = X_train_with_outliers['num_providers'].mean()
        temperature_mean = X_train_with_outliers['temperature_2m_mean'].mean()

        selected_state = request.form['state']
        states_vector = [1 if state.lower() == selected_state.lower().strip() else 0 for state in states_list]
        temperature = float(request.form['temperature'])
        num_ili = float(request.form['num_ili'])
        num_providers = float(request.form['num_providers'])

        features = [epiweek_current, temperature, num_ili, num_providers] + states_vector
        print(f"Valores ingresados: epiweek={epiweek_current}, Temperature={temperature}, Num_ILI={num_ili}, Num_Providers={num_providers}, States={states_vector}")
        features_df = pd.DataFrame([features], columns=feature_columns)
        prediction = model.predict(features_df)

        return render_template(
            "form.html",
            prediction=f"{prediction[0]:.2f}",
            state=selected_state,
            default_num_ili=default_num_ili,
            default_num_providers=default_num_providers,
            temperature_mean=temperature_mean,
            states_list=states_list
        )
    else:
        # Para GET simplemente muestra el formulario sin hacer validaciones.
        return render_template(
            "form.html",
            default_num_ili=X_train_with_outliers['num_ili'].mean(),
            default_num_providers=X_train_with_outliers['num_providers'].mean(),
            temperature_mean=X_train_with_outliers['temperature_2m_mean'].mean(),
            states_list=states_list
        )

if __name__ == "__main__":
    app.run(debug=True)
