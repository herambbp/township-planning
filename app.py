
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template
import requests
import json
from google import genai


app = Flask(__name__)

# Global variables to store models and data processors
voting_clf = None
scaler = None
label_encoder = None
final_features = None


def load_models():
    """
    Load models from disk
    """
    global voting_clf, scaler, label_encoder, final_features

    print("Loading models...")
    try:

        with open('models/voting_clf.pkl', 'rb') as f:
            voting_clf = pickle.load(f)
            print("voting done")

        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            print("scalar done")

        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            print("label done")

        with open('models/final_features.pkl', 'rb') as f:
            final_features = pickle.load(f)
            print("final done")

        print("Models loaded successfully!")
        return True
    except FileNotFoundError:
        print("Models not found. Please train models first.")
        return False


def predict_township_feasibility(city_data, model=None, scaler=None, features=None, encoder=None):
    """
    Predict township feasibility for new city data

    Parameters:
    city_data : dict - Dictionary containing city features
    model : trained model to use for prediction
    scaler : fitted scaler to standardize the features
    features : list of features used by the model
    encoder : label encoder to decode predictions

    Returns:
    prediction : str - Predicted feasibility
    probabilities : dict - Probability of each class
    """
    # Use global variables if not provided
    model = model or voting_clf
    scaler = scaler or globals()['scaler']
    features = features or final_features
    encoder = encoder or label_encoder

    # Calculate derived features if not present
    if 'Population_Density' not in city_data and 'Population' in city_data and 'Area (sq. km)' in city_data:
        city_data['Population_Density'] = city_data['Population'] / \
            city_data['Area (sq. km)']

    if 'Infrastructure_per_Person' not in city_data and 'Smart Infrastructure Score' in city_data and 'Population' in city_data:
        city_data['Infrastructure_per_Person'] = city_data['Smart Infrastructure Score'] / \
            city_data['Population'] * 1000

    if 'Transport_Air_Quality_Ratio' not in city_data and 'Public Transport Usage' in city_data and 'Air Quality Index' in city_data:
        city_data['Transport_Air_Quality_Ratio'] = city_data['Public Transport Usage'] / \
            (city_data['Air Quality Index'] + 1)

    if 'Smart_Living_Index' not in city_data:
        indices = ['Smart Infrastructure Score',
                   'Education Index', 'Healthcare Index', 'Safety Index']
        if all(idx in city_data for idx in indices):
            city_data['Smart_Living_Index'] = sum(
                city_data[idx] for idx in indices) / 4

    if 'Resource_Efficiency' not in city_data and 'Waste Management Score' in city_data and 'Energy Consumption' in city_data:
        city_data['Resource_Efficiency'] = (
            city_data['Waste Management Score'] / (city_data['Energy Consumption'] + 1)) * 100

    # Create DataFrame with only the required features
    df_pred = pd.DataFrame([city_data])

    # Check for missing features
    missing_features = [f for f in features if f not in df_pred.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    df_pred = df_pred[features]

    # Scale the features
    df_pred_scaled = scaler.transform(df_pred)

    # Make prediction
    pred_class = model.predict(df_pred_scaled)[0]
    pred_proba = model.predict_proba(df_pred_scaled)[0]

    # Decode prediction
    prediction = encoder.inverse_transform([pred_class])[0]

    # Get probabilities for each class
    probabilities = {cls: float(prob)
                     for cls, prob in zip(encoder.classes_, pred_proba)}

    return prediction, probabilities


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Check if models are loaded
    if voting_clf is None or scaler is None or label_encoder is None or final_features is None:
        return jsonify({"error": "Models not loaded. Please initialize the app correctly."}), 500

    # Get data from request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Make prediction
        prediction, probabilities = predict_township_feasibility(data)

        # Prepare response
        response = {
            "prediction": prediction,
            "probabilities": probabilities
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/features', methods=['GET'])
def get_features():
    """Return the list of features used by the model"""
    if final_features is None:
        return jsonify({"error": "Models not loaded. Please initialize the app correctly."}), 500

    return jsonify({
        "required_features": final_features,
        "derived_features": [
            "Population_Density",
            "Infrastructure_per_Person",
            "Transport_Air_Quality_Ratio",
            "Smart_Living_Index",
            "Resource_Efficiency"
        ]
    })

# Create a sample dataset if one doesn't exist


def create_sample_data(filename='smart_city_data.csv'):
    """
    Create a sample dataset if one doesn't exist
    """
    if os.path.exists(filename):
        print(f"Using existing data file: {filename}")
        return

    print(f"Creating sample data file: {filename}")

    # Define features
    features = [
        'Population', 'Area (sq. km)', 'Smart Infrastructure Score',
        'Energy Consumption', 'Public Transport Usage', 'Air Quality Index',
        'Education Index', 'Healthcare Index', 'Employment Rate',
        'Smart Grid Adoption', 'Waste Management Score', 'Internet Speed (Mbps)',
        'Safety Index', 'Cost of Living Index'
    ]

    # Generate 200 random cities
    np.random.seed(42)
    n_samples = 200

    data = {
        'Population': np.random.randint(100000, 15000000, n_samples),
        'Area (sq. km)': np.random.randint(50, 2000, n_samples),
        'Smart Infrastructure Score': np.random.randint(20, 95, n_samples),
        'Energy Consumption': np.random.randint(200, 1200, n_samples),
        'Public Transport Usage': np.random.randint(10, 90, n_samples),
        'Air Quality Index': np.random.randint(10, 100, n_samples),
        'Education Index': np.random.randint(30, 95, n_samples),
        'Healthcare Index': np.random.randint(30, 95, n_samples),
        'Employment Rate': np.random.randint(30, 90, n_samples),
        'Smart Grid Adoption': np.random.randint(10, 90, n_samples),
        'Waste Management Score': np.random.randint(20, 90, n_samples),
        'Internet Speed (Mbps)': np.random.randint(50, 1000, n_samples),
        'Safety Index': np.random.randint(30, 90, n_samples),
        'Cost of Living Index': np.random.randint(30, 150, n_samples)
    }

    df = pd.DataFrame(data)

    # Add derived features to help with classification
    df['Population_Density'] = df['Population'] / df['Area (sq. km)']
    df['Infrastructure_per_Person'] = df['Smart Infrastructure Score'] / \
        df['Population'] * 1000
    df['Transport_Air_Quality_Ratio'] = df['Public Transport Usage'] / \
        (df['Air Quality Index'] + 1)
    df['Smart_Living_Index'] = (df['Smart Infrastructure Score'] + df['Education Index'] +
                                df['Healthcare Index'] + df['Safety Index']) / 4
    df['Resource_Efficiency'] = (
        df['Waste Management Score'] / (df['Energy Consumption'] + 1)) * 100

    # Define a complex rule for township feasibility
    conditions = [
        (df['Smart_Living_Index'] > 75) & (df['Population_Density']
                                           < 10000) & (df['Resource_Efficiency'] > 15),
        (df['Smart_Living_Index'] > 60) & (df['Population_Density']
                                           < 5000) & (df['Resource_Efficiency'] > 10),
        (df['Transport_Air_Quality_Ratio'] > 1.5) & (
            df['Infrastructure_per_Person'] > 0.015)
    ]
    choices = ['Highly Feasible',
               'Moderately Feasible', 'Potentially Feasible']
    df['Township Feasibility'] = np.select(conditions, choices, 'Not Feasible')

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")


def clean_gemini_response(text):
    # Remove ```json and ```
    return text.strip().removeprefix("```json").removesuffix("```").strip()


# genai.configure(api_key="AIzaSyDiO3SFDITVn3hktp53mnlFBl-c2Ucipfk")
genai_client = genai.Client(api_key="AIzaSyDiO3SFDITVn3hktp53mnlFBl-c2Ucipfk")


@app.route('/fetch-data', methods=['POST'])
def gemini_generate_and_predict():
    city = request.form.get('city')
    
    prompt = f"""
Provide **realistic, varied, and city-specific** JSON data for urban parameters of {city}.
Do **not return ideal or overly optimistic values**. Base the values on plausible estimates
based on the city's region, development level, and known characteristics. Use diverse values
across cities (e.g., a small city shouldn't have the same values as a tech capital).

Output format:
{{
    "Population": number (in millions),
    "Area (sq. km)": number,
    "Smart Infrastructure Score": number (0–100),
    "Energy Consumption": number (kWh per capita),
    "Public Transport Usage": number (percentage of population),
    "Air Quality Index": number (0–500, lower is better),
    "Education Index": number (0.0–1.0),
    "Healthcare Index": number (0.0–1.0),
    "Employment Rate": number (percentage),
    "Smart Grid Adoption": number (percentage),
    "Waste Management Score": number (0–100),
    "Internet Speed (Mbps)": number,
    "Safety Index": number (0–100),
    "Cost of Living Index": number (0–100)
}}

Consider the following when generating data for {city}:
- Its approximate population (e.g., megacity, large, medium, small).
- Its level of economic development (e.g., high, upper-middle, lower-middle, low income).
- Its primary economic sectors (e.g., technology, manufacturing, agriculture, tourism).
- Its geographical location and any known environmental factors.

**Crucially, ensure the values are plausible and reflect the likely realities of a city like {city}. Avoid consistently high scores across all parameters.**

Example Guidance:
- For a major developed city known as a tech hub (e.g., San Francisco): Expect high Smart Infrastructure, high Internet Speed, high Cost of Living, potentially moderate Public Transport Usage, and varying scores on other indices based on specific local conditions.
- For a large developing city with significant industrial activity (e.g., Mumbai): Expect a large Population, potentially moderate Smart Infrastructure, higher Energy Consumption relative to efficiency, high Public Transport Usage, potentially higher Air Quality Index, and a Cost of Living that reflects its economic activity.
- For a smaller, less developed city focused on agriculture (e.g., a rural capital in sub-Saharan Africa): Expect a lower Population, potentially lower scores across most infrastructure and technology indices, lower Energy Consumption, potentially lower Cost of Living, and varying scores on social indices based on local investment.
"""

    # Get Gemini's response
    gemini_response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    try:
        
        data = clean_gemini_response(gemini_response.text);
        city_data = json.loads(data)
    except json.JSONDecodeError:
        return "Gemini didn't return valid JSON. Try again." + gemini_response.text, 500

    # Forward to /predict
    prediction_response = requests.post("http://localhost:5000/predict", json=city_data)

    if prediction_response.status_code == 200:
        prediction = prediction_response.json()
        return render_template("result.html", city=city, prediction=prediction)
    else:
        return "Prediction failed", 500


# Main function to initialize the app
if __name__ == '__main__':
    # Create sample data if needed
    create_sample_data()

    # Try to load models, train if not available
    if not load_models():
        print("Models not found")

    # Run the Flask app
    app.run(debug=True)
