import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'credit_card_approval_secret_key'  # Used for flash validation messages

# Define paths for model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
MODEL_NAME_PATH = os.path.join(BASE_DIR, 'model', 'best_model_name.txt')

# Load the serialised artifacts
model = None
best_model_name = "XGBoost (Default)"

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[SERVER INFO] Model loaded successfully.")
    else:
        print("[SERVER WARNING] Model file not found.")

    if os.path.exists(MODEL_NAME_PATH):
        with open(MODEL_NAME_PATH, 'r') as file:
            best_model_name = file.read().strip()
        print(f"[SERVER INFO] Selected model name: {best_model_name}")
except Exception as e:
    print(f"[SERVER ERROR] Error during initialization of artifacts: {e}")

@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('home.html', model_name=best_model_name)

@app.route('/predict', methods=['GET'])
def predict_form():
    """Renders the evaluation form."""
    return render_template('index.html')

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    """Processes form submission and predicts approval status."""
    if request.method == 'POST':
        try:
            form_data = request.form
            
            # Extract inputs
            age_raw = form_data.get('age')
            income_raw = form_data.get('income')
            debt_raw = form_data.get('debt')
            credit_score_raw = form_data.get('credit_score')
            
            # Validation checks
            if not all([age_raw, income_raw, debt_raw, credit_score_raw]):
                flash("All fields are required. Please check your inputs.", "danger")
                return redirect(url_for('predict_form'))
                
            try:
                age = int(age_raw)
                income = float(income_raw)
                debt = float(debt_raw)
                credit_score = int(credit_score_raw)
            except ValueError:
                flash("Invalid input format. Age and Credit Score must be integers, and Income and Debt must be numbers.", "danger")
                return redirect(url_for('predict_form'))
                
            if age < 18 or age > 100:
                flash("Age must be between 18 and 100.", "danger")
                return redirect(url_for('predict_form'))
                
            if income < 0 or debt < 0:
                flash("Income and Debt cannot be negative values.", "danger")
                return redirect(url_for('predict_form'))
                
            if credit_score < 300 or credit_score > 850:
                flash("Credit Score must be between 300 and 850.", "danger")
                return redirect(url_for('predict_form'))
                
            # Preprocessing:
            # Map credit score of 300-850 to 0-30 scale for Total_Good_Debt
            scaled_credit_score = ((credit_score - 300) / (850 - 300)) * 30.0

            # Arrange inputs in the exact sequence expected by the model
            raw_features = [income, age, debt, scaled_credit_score]
            
            # Predict using model
            if model is not None and os.path.exists(SCALER_PATH):
                # Import preprocessing module helpers to handle scaling cleanly
                from preprocessing import load_scaler_and_transform
                scaled_features = load_scaler_and_transform(raw_features, SCALER_PATH)
                
                # Make prediction
                prediction = model.predict(scaled_features)[0]
                print(f"[SERVER DEBUG] Scaled Input: {scaled_features} | Raw Model Prediction: {prediction}")
            else:
                # Fallback rule-based logic if model files are missing
                print("[SERVER WARNING] Running rule-based prediction fallback.")
                if credit_score >= 600 and debt / max(income, 1) < 0.4:
                    prediction = 1
                else:
                    prediction = 0
            
            # Map prediction: 1 = Approved, 0 = Rejected
            if prediction == 1:
                result_text = "Approved"
            else:
                result_text = "Rejected"
            
            # Calculate credit risk metrics to display in Bootstrap UI
            debt_to_income = (debt / max(income, 1)) * 100
            
            return render_template(
                'result.html', 
                prediction_result=result_text,
                model_name=best_model_name,
                age=age,
                income=income,
                debt=debt,
                credit_score=credit_score,
                debt_to_income=f"{debt_to_income:.1f}%"
            )
            
        except Exception as e:
            return f"An error occurred during prediction processing: {str(e)}", 500

    return redirect(url_for('predict_form'))

if __name__ == '__main__':
    app.run(debug=True)