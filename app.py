from flask import Flask, render_template, request, jsonify


from newPython import get_response

 
app = Flask(__name__)

from flask import send_from_directory

@app.route('/intents.json')
def serve_intents():
    return send_from_directory(r'c:\Users\LENOVO\anaconda folder\envs\myenv\Lib\site-packages\tflearn', 'intents.json')



@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text) 
    message = {"answer":response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)