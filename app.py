from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Define the expected features and their order
EXPECTED_FEATURES = [
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
]

def load_model():
    """Load the trained model and scaler"""
    try:
        # loading the model 
        model_path = os.path.join('models', 'lgbm_model.pkl')
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # loading standard scaler model
        scaler_path = os.path.join('data', 'models', 'standard_scaler.pkl')
        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        raise

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        input_df = pd.DataFrame([{feature: data.get(feature, 0) for feature in EXPECTED_FEATURES}])
        logger.info(f"Input DataFrame shape: {input_df.shape}")
        
        model, scaler = load_model()
        
        scaled_features = scaler.transform(input_df)
        logger.info("Features scaled successfully")
        prediction = model.predict(scaled_features)[0]
        logger.info(f"Prediction result: {prediction}")
        return jsonify({'success': True, 'prediction': float(prediction)})
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    try:
        model, scaler = load_model()
        logger.info("Model loaded successfully, starting Flask app")
        app.run(debug=True, port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")