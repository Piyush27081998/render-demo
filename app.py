from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained logistic regression model
model_path = "model.pkl"
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print("Error loading model:", e)
        model = None
else:
    print("Model file not found!")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        self_employed = 1 if data["selfEmployed"] == "Yes" else 0
        previous_loan = 1 if data["previousLoan"] == "Yes" else 0
        age = float(data["age"])
        cibil_score = float(data["cibilScore"])
        tenure = float(data["tenure"])

        # Convert input to array format
        input_data = np.array([[self_employed, previous_loan, age, cibil_score, tenure]])

        # Ensure model exists before prediction
        if model:
            prediction = model.predict(input_data)  # Ensure input shape is correct
            result = "Approved" if prediction[0] == 1 else "Not Approved"
        else:
            result = "Model not loaded"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
