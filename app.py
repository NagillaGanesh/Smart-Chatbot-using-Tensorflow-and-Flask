from flask import Flask, render_template, request, jsonify, send_from_directory
import os
#from newPython import get_response
from newPythonSpare import get_response

app = Flask(__name__)

@app.route('/intents.json')
def serve_intents():
    # Dynamically serve intents.json from the current working directory
    return send_from_directory(os.getcwd(), 'intents.json')

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)

#chatbot38-env\Scripts\activate
#http://127.0.0.1:5000/