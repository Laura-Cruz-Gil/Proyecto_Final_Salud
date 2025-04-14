from flask import Flask, redirect, request, render_template, url_for
import joblib
import numpy as np
import os
import pandas as pd
from datetime import date

model_path = os.path.join(os.path.dirname(__file__), '../models/best_random_forest_model_with_regions.joblib')
model = joblib.load(model_path)

y_train = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/processed/y_with_regions_train.csv'), delimiter=',')
X_train_with_outliers = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/processed/X_train_with_regions_with_outliers_std.csv'), delimiter=',')

TEMPERATURE_MIN = -62.2
TEMPERATURE_MAX = 56.7

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


state_to_hhs = {
    "alabama": "region_HHS4", "alaska": "region_HHS10", "arizona": "region_HHS9", "arkansas": "region_HHS6",
    "california": "region_HHS9", "colorado": "region_HHS8", "connecticut": "region_HHS1",
    "delaware": "region_HHS3", "florida": "region_HHS4", "georgia": "region_HHS4", "hawaii": "region_HHS9",
    "idaho": "region_HHS10", "illinois": "region_HHS5", "indiana": "region_HHS5", "iowa": "region_HHS7",
    "kansas": "region_HHS7", "kentucky": "region_HHS4", "louisiana": "region_HHS6", "maine": "region_HHS1",
    "maryland": "region_HHS3", "massachusetts": "region_HHS1", "michigan": "region_HHS5", "minnesota": "region_HHS5",
    "mississippi": "region_HHS4", "missouri": "region_HHS7", "montana": "region_HHS8", "nebraska": "region_HHS7",
    "nevada": "region_HHS9", "new hampshire": "region_HHS1", "new jersey": "region_HHS2", "new mexico": "region_HHS6",
    "new york": "region_HHS2", "north carolina": "region_HHS4", "north dakota": "region_HHS8", "ohio": "region_HHS5",
    "oklahoma": "region_HHS6", "oregon": "region_HHS10", "pennsylvania": "region_HHS3", "rhode island": "region_HHS1",
    "south carolina": "region_HHS4", "south dakota": "region_HHS8", "tennessee": "region_HHS4", "texas": "region_HHS6",
    "utah": "region_HHS8", "vermont": "region_HHS1", "virginia": "region_HHS3", "washington": "region_HHS10",
    "west virginia": "region_HHS3", "wisconsin": "region_HHS5", "wyoming": "region_HHS8"
}

region_cols = [f"region_HHS{i}" for i in range(1, 11)]
feature_columns = ["epiweek", "temperature_2m_mean", "num_ili", "num_providers"] + region_cols


def validate_request_data(request) -> str:
    """ Validates the request data for the prediction endpoint. """
    if not request.form.get('num_ili') or not request.form.get('num_providers'):
        return "Error: All fields are required (num_ili, num_providers)."
    try:
        float(request.form['num_ili'])
        float(request.form['num_providers'])
    except ValueError:
        return "Error: Invalid input. Please enter numeric values for num_ili and num_providers."

    if request.form.get('temperature'):
        try:
            temperature = float(request.form['temperature'])
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
        error_message = validate_request_data(request)
        if error_message:
            return error_message

        today = date.today()
        year, week, _ = today.isocalendar()
        epiweek_current = int(f"{year}{week:02d}")

        selected_state = request.form['state']
        region = state_to_hhs[selected_state.lower()]
        regions_vector = [1 if col == region else 0 for col in region_cols]

        
        if not request.form.get('temperature'):
            temperature = X_train_with_outliers['temperature_2m_mean'].mean()
        else:
            temperature = float(request.form['temperature'])
            
        num_ili = float(request.form['num_ili'])
        num_providers = float(request.form['num_providers'])

        features = [epiweek_current, temperature, num_ili, num_providers] + regions_vector
        print(f"Features: {features}")
        features_df = pd.DataFrame([features], columns=feature_columns)

        features_df = features_df[model.feature_names_in_]

        prediction = model.predict(features_df)

        return render_template(
            "form.html",
            prediction=f"{prediction[0]:.2f}",
            region=region.replace("region_", ""),
            state=selected_state,
            default_num_ili=num_ili,
            default_num_providers=num_providers,
            temperature_mean=temperature,
            states_list=states_list
        )
    else:
        return render_template(
            "form.html",
            default_num_ili=X_train_with_outliers['num_ili'].max(),
            default_num_providers=X_train_with_outliers['num_providers'].max(),
            temperature_mean=X_train_with_outliers['temperature_2m_mean'].max(),
            states_list=states_list
        )


if __name__ == "__main__":
    app.run(debug=True)
