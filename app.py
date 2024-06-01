from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)
model = load('models/best_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    predictions = [0 if x == 1 else 1 for x in predictions]
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)