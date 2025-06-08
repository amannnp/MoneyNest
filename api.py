from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app) 

# model and encoder
model = joblib.load('loan_default_model.pkl')
encoder = joblib.load('loan_default_encoder.pkl')

# training cols
cat_cols = ['Loan_Type', 'Gender', 'Marital_Status']
num_cols = ['Age', 'Credit_Score', 'Annual_Income_INR', 'Loan_Term_Months']

@app.route("/")
def home():
    return "MoneyNest Loan Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received JSON:", data)

        # Prepare raw data dict with correct column names
        input_dict = {
            'Age': [float(data['Age'])],
            'Credit_Score': [float(data['Credit_Score'])],
            'Annual_Income_INR': [float(data['Annual_Income_INR'])],
            'Loan_Term_Months': [float(data['Loan_Term_Months'])],
            'Loan_Type': [data['Loan_Type']],
            'Gender': [data['Gender']],
            'Marital_Status': [data['Marital_Status']]
        }

        input_df = pd.DataFrame(input_dict)
        encoded_cat = encoder.transform(input_df[cat_cols])
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))
        final_input = pd.concat([input_df[num_cols].reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

        print("Final input to model:", final_input)

        # prediction
        prediction = model.predict(final_input)[0]
        return jsonify({
            "prediction": int(prediction),
            "message": "Loan eligibility prediction successful."
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400
# debug and port define
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
