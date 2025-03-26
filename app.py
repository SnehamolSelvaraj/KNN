from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping for labels
quality_map = {0: "Low", 1: "Medium", 2: "High"}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        alcohol = float(request.form['alcohol'])
        pH = float(request.form['pH'])

        # Prepare input for model
        input_features = np.array([[alcohol, pH]])
        input_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_quality = quality_map[prediction[0]]

        return render_template("index.html", prediction=f"Predicted Wine Quality: {predicted_quality}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
