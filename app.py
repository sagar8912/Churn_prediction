from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from utils import preprocess_input

app = Flask(__name__)

# Load the model
with open('Chunk.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form.to_dict()
    
    # Convert the data to a DataFrame
    input_data = pd.DataFrame([data])
    
    # Preprocess the data
    processed_data = preprocess_input(input_data)
    
    # Make the prediction
    prediction = model.predict(processed_data)
    prediction_text = 'Yes' if prediction[0] == 1 else 'No'
    
    # Return the prediction as JSON
    return jsonify({'Exited': int(prediction[0]), 'Prediction': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)

