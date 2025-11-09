import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    data = request.json['data']
    print("Input JSON:", data)

    input_data = np.array(list(data.values())).reshape(1, -1)
    print("Input array:", input_data)

    new_data = scaler.transform(input_data)
    output = regmodel.predict(new_data)
    print("Prediction:", output[0])

    return jsonify({'prediction': float(output[0])})

if __name__ == "__main__":
    app.run(debug=True)
