from flask import Flask, request, render_template, redirect, url_for
from joblib import load
from nltk.tokenize import TweetTokenizer
import pandas as pd
from static.tokenize import tokenize


app = Flask(__name__)

# main route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/analisis', methods=['GET', 'POST'])
def analisis():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('analisis.html')

@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('info.html')

@app.route("/predict", methods=["POST"])
def predict_function():
    if request.method == "POST":
        model_path = open("./model/model_rf.joblib", "rb")
        model = load(model_path)
        input_payloads = request.json["inputs"]
        predictions = model.predict(input_payloads)
        json_result = pd.Series(predictions).to_json(orient="values")
        return json_result

###
# this function is required by the model as it needs the text to be tokenized first with the same tokenization algorithm
# before supplied to the machine learning model
def tokenize(text):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)


if __name__ == "__main__":
    app.run(debug=True)
