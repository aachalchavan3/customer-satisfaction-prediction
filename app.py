from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "customer_satisfaction_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        data = {
            "Age": int(request.form["Age"]),
            "Gender": request.form["Gender"],
            "Service_Quality": int(request.form["Service_Quality"]),
            "Product_Quality": int(request.form["Product_Quality"]),
            "Support_Quality": int(request.form["Support_Quality"]),
            "Waiting_Time": int(request.form["Waiting_Time"]),
            "Value_for_Money": int(request.form["Value_for_Money"]),
            "Ease_of_Use": int(request.form["Ease_of_Use"]),
            "Purchase_Frequency": int(request.form["Purchase_Frequency"]),
            "Recommendation": int(request.form["Recommendation"]),
            "Complaint_Resolution": int(request.form["Complaint_Resolution"])
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability = round(float(model.predict_proba(input_df).max()) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
