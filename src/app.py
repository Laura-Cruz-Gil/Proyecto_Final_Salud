from flask import Flask, request, render_template
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
    return "¡Bienvenido a la app de predicción de la gripe!"

@app.route('/form', methods=['GET', 'POST'])
def form():

    error_message = validate_request_data(request)
    if error_message:
        return error_message
    
    today = date.today()
    year, week, _ = today.isocalendar()
    epiweek_current = int(f"{year}{week:02d}") 

    default_num_ili = X_train_with_outliers['num_ili'].mean()
    default_num_providers = X_train_with_outliers['num_providers'].mean()
    temperature_mean = X_train_with_outliers['temperature_2m_mean'].mean()

    if request.method == 'POST':

        selected_state = request.form['state']
        
        states_vector = [1 if state.lower() == selected_state.lower().strip() else 0 for state in states_list]

        temperature = float(request.form['temperature'])

        num_ili = float(request.form['num_ili'])
        num_providers = float(request.form['num_providers'])

        features = [epiweek_current, temperature, num_ili, num_providers] + states_vector
        print(f"Valores ingresados: epiweek_default={epiweek_current:.2f}, Temperature={temperature}, Num_ILI={num_ili}, Num_Providers={num_providers}, States={states_vector}")
        
        features_df = pd.DataFrame([features], columns=feature_columns)
        prediction = model.predict(features_df)
        
        explanation = f"""
            Predicción para {selected_state.title()}: {prediction[0]:.2f}%<br>
            Esto representa el porcentaje de visitas médicas relacionadas con síntomas similares
            a los de la gripe en el estado seleccionado.<br>
            <br>
            **Nota:** Si no se proporcionan entradas reales, los resultados podrían ser menos precisos.
        """
        return explanation
    
    return f'''
        <form method="POST">
            <label for="num_ili">Número de visitas médicas relacionadas con la gripe:</label>
            <input type="text" name="num_ili" id="num_ili" required placeholder="{default_num_ili:.2f}"><br>
            
            <label for="num_providers">Número de proveedores médicos:</label>
            <input type="text" name="num_providers" id="num_providers" required placeholder="{default_num_providers:.2f}"><br>
            
            <label for="temperature">Temperatura promedio (°C):</label>
            <input type="text" name="temperature" id="temperature" placeholder="{temperature_mean:.2f}"><br>
            
            <label for="state">Selecciona un estado:</label>
            <select name="state" id="state" required>
                {''.join([f'<option value="{state}">{state.title()}</option>' for state in states_list])}
            </select><br>
            
            <button type="submit">Predecir</button>
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
