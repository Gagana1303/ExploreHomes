from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained XGBoost pipeline model
model = joblib.load("models/bangalore_xgb_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/contact")
def contact_page():
    return render_template("contact.html")

@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json

        # Extract OneHotEncoder from pipeline
        ohe = model.named_steps["preprocess"].transformers_[0][1]
        trained_locations = list(ohe.categories_[0])

        # Location cleaning function
        def clean_location(loc: str):
            loc = loc.strip().title()
            replacements = {
                "White Field": "Whitefield",
                "Marathalli": "Marathahalli",
                "Hsrlayout": "Hsr Layout",
                "Hsr Layout ": "Hsr Layout",
            }
            return replacements.get(loc, loc)

        # Clean user location
        loc = clean_location(data["location"])

        # If location not recognized → map to "Other"
        if loc not in trained_locations:
            loc = "Other"

        # Build input DF
        user_df = pd.DataFrame([[
            loc,
            float(data["total_sqft"]),
            int(data["bath"]),
            int(data["bhk"])
        ]], columns=["location", "total_sqft", "bath", "bhk"])

        # Predict
        predicted_price = float(model.predict(user_df)[0])

        return jsonify({"estimated_price_lakhs": round(predicted_price, 2)})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 400

@app.route("/send-contact", methods=["POST"])
def send_contact():
    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    message = data.get("message")

    if not name or not email or not message:
        return jsonify({"success": False})

    try:
        with open("contact_messages.txt", "a", encoding="utf-8") as file:
            file.write("----- New Message -----\n")
            file.write(f"Time: {datetime.now()}\n")
            file.write(f"Name: {name}\n")
            file.write(f"Email: {email}\n")
            file.write("Message:\n")
            file.write(message + "\n")
            file.write("-----------------------\n\n")

        return jsonify({"success": True})

    except Exception as e:
        print(e)
        return jsonify({"success": False})


if __name__ == "__main__":
    app.run(debug=True)
