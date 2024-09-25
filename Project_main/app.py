from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and preprocessors
with open('dtr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('ohe.pkl', 'rb') as ohe_file:
    ohe = pickle.load(ohe_file)

with open('scale.pkl', 'rb') as scale_file:
    scale = pickle.load(scale_file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        Crop_Year = float(request.form['Crop_Year'])
        Area = float(request.form['Area'])  # Only Crop_Year and Area are used
        Season = request.form['Season']
        State = request.form['State']
        Crop = request.form['Crop']
        
        # Prepare the numeric features (without Annual_Rainfall, Fertilizer, Pesticide)
        numeric_features = np.array([[Crop_Year, Area]], dtype=float)
        
        # Prepare the categorical features
        categorical_features = np.array([[Season, State, Crop]])
        
        # Scale numeric features
        scaled_numeric_features = scale.transform(numeric_features)
        
        # One-hot encode categorical features
        encoded_categorical_features = ohe.transform(categorical_features)
        
        # Combine all features
        transform_features = np.hstack([scaled_numeric_features, encoded_categorical_features])
        
        # Predict using the trained model
        predicted_yield = model.predict(transform_features)
        
        return render_template('result.html', predicted_yield=predicted_yield[0])
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
