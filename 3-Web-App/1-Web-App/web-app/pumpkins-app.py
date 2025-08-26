import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("../../../2-Regression/4-Logistic/pumpkins-model.pkl", "rb"))
preprocessor = pickle.load(
    open("../../../2-Regression/4-Logistic/pumpkins-preprocessor.pkl", "rb")
)
label_encoder = pickle.load(
    open("../../../2-Regression/4-Logistic/pumpkins-label-encoder.pkl", "rb")
)


@app.route("/")
def home():
    return render_template("pumpkins-index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "City Name": request.form["City"],
        "Package": request.form["Package"],
        "Variety": request.form["Variety"],
        "Origin": request.form["Origin"],
        "Item Size": request.form["Size"],
    }

    columns_order = ["City Name", "Package", "Variety", "Origin", "Item Size"]
    df = pd.DataFrame([input_data])[columns_order]

    X_transformed = preprocessor.transform(df)

    pred_encoded = model.predict(X_transformed.to_numpy())[0]
    pred_color = label_encoder.inverse_transform([pred_encoded])[0]

    return render_template(
        "pumpkins-index.html",
        prediction_text="Predicted pumpkin color: {}".format(pred_color),
    )


if __name__ == "__main__":
    app.run(debug=True)
