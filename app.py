from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
import datetime
import logging

# Enhanced logging setup
logging.basicConfig(
    filename='fraud_detection_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Load the saved models and thresholds
def load_models(model_dir='models', use_smote=True):
    prefix = 'smote_' if use_smote else ''
    
    try:
        # Load ensemble model
        model_path = os.path.join(model_dir, f'{prefix}ensemble_model.pkl')
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            ensemble_model = pickle.load(f)
            print(f"Loaded {prefix}ensemble_model.pkl")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f'{prefix}scaler.pkl')
        if not os.path.exists(scaler_path):
            logging.error(f"Scaler file not found: {scaler_path}")
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            print(f"Loaded {prefix}scaler.pkl")
        
        # Load thresholds if available
        thresholds = {'default': 0.5, 'conservative': 0.01}  # Default if no threshold file
        
        threshold_path = os.path.join(model_dir, f'{prefix}thresholds.pkl')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'rb') as f:
                loaded_thresholds = pickle.load(f)
                thresholds.update(loaded_thresholds)
                print(f"Loaded {prefix}thresholds.pkl")
        
        # Step 2: Print all threshold values for verification
        print("Threshold values:")
        for key, value in thresholds.items():
            print(f"  {key}: {value}")
            
        # Force conservative threshold to be very sensitive (Step 3)
        thresholds['conservative'] = 0.01  # 1% threshold for high sensitivity
        print(f"Forced 'conservative' threshold to {thresholds['conservative']} for maximum sensitivity")
        
        return ensemble_model, scaler, thresholds
    
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Initialize models
try:
    print("Loading SMOTE-trained models...")
    ensemble_model, scaler, thresholds = load_models(use_smote=True)
    print("Models loaded successfully")
    print(f"Available thresholds: {list(thresholds.keys())}")
    
    # Use F1-optimized threshold by default
    default_threshold = thresholds.get('f1', 0.5)
    print(f"Using default threshold: {default_threshold}")
    
except Exception as e:
    print(f"Error initializing models: {e}")
    # Use dummy models if loading fails (for development)
    ensemble_model = None
    scaler = None
    thresholds = {'default': 0.5, 'conservative': 0.01}
    default_threshold = 0.5

@app.route('/')
def home():
    return render_template('index.html', threshold_options=thresholds)

@app.route('/predict', methods=['POST'])
def predict():
    transaction_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    logging.info(f"[{transaction_id}] New prediction request received")
    
    try:
        # For API calls
        if request.is_json:
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
            
            # Get custom threshold if provided
            threshold_key = data.get('threshold_key', 'f1')
            threshold = thresholds.get(threshold_key, default_threshold)
            
            logging.info(f"[{transaction_id}] API request with {features.shape[1]} features, threshold_key: {threshold_key}")
        else:
            # For form submissions
            features = []
            # Get all features from form
            for i in range(30):  # Changed to handle 30 features
                feature_name = f'feature_{i}'
                if feature_name in request.form:
                    features.append(float(request.form[feature_name]))
            
            features = np.array(features).reshape(1, -1)
            
            # Get threshold selection from form if available
            threshold_key = request.form.get('threshold_option', 'f1')
            threshold = thresholds.get(threshold_key, default_threshold)
            
            # Step 2: Log the actual threshold values
            logging.info(f"[{transaction_id}] Threshold options: {thresholds}")
            logging.info(f"[{transaction_id}] Selected threshold '{threshold_key}': {threshold}")
        
        # Step 4: Log raw feature values for debugging
        logging.info(f"[{transaction_id}] Raw features (first 5): {features[0, :5]}")
        
        # Check if models are loaded
        if ensemble_model is None or scaler is None:
            error_msg = "Models not loaded. Please check server logs."
            logging.error(f"[{transaction_id}] {error_msg}")
            return jsonify({"error": error_msg, "transaction_id": transaction_id}), 500
        
        # Check feature count and pad if necessary
        expected_features = scaler.mean_.shape[0]  # Should be 30
        actual_features = features.shape[1]        # Currently 30
        
        # Handle feature count mismatch by padding with zeros
        if actual_features < expected_features:
            logging.warning(f"[{transaction_id}] Feature count mismatch. Padding {actual_features} to {expected_features} features")
            padding = np.zeros((1, expected_features - actual_features))
            features = np.hstack((features, padding))
            logging.info(f"[{transaction_id}] Features padded to {features.shape[1]} features")
        elif actual_features > expected_features:
            logging.warning(f"[{transaction_id}] Too many features. Truncating from {actual_features} to {expected_features}")
            features = features[:, :expected_features]
        
        # Scale features
        try:
            scaled_features = scaler.transform(features)
            # Step 4: Log scaled feature values for debugging
            logging.info(f"[{transaction_id}] Scaled features (first 5): {scaled_features[0, :5]}")
        except Exception as e:
            error_msg = f"Error scaling features: {str(e)}"
            logging.error(f"[{transaction_id}] {error_msg}")
            return jsonify({"error": error_msg, "transaction_id": transaction_id}), 500
        
        # Make prediction
        try:
            prediction_proba = ensemble_model.predict_proba(scaled_features)
            original_fraud_probability = float(prediction_proba[0][1])
            legitimate_probability = float(prediction_proba[0][0])
            
            # Step 6: Apply probability adjustment to make model more sensitive to fraud
            # Multiply fraud probability by 3 (adjust this factor as needed)
            adjustment_factor = 10.0
            adjusted_fraud_probability = min(original_fraud_probability * adjustment_factor, 1.0)
            
            # Log both probabilities
            logging.info(f"[{transaction_id}] Original fraud probability: {original_fraud_probability:.4f}")
            logging.info(f"[{transaction_id}] Adjusted fraud probability (x{adjustment_factor}): {adjusted_fraud_probability:.4f}")
            
            # Use BOTH the selected threshold AND the adjusted probability
            prediction = 1 if adjusted_fraud_probability >= threshold else 0
            
            # Step 3: Override with a direct threshold for testing if using 'conservative' setting
            if threshold_key == 'conservative':
                # Use an aggressive fixed threshold of 0.01 (1%)
                direct_threshold = 0.01
                direct_prediction = 1 if original_fraud_probability >= direct_threshold else 0
                logging.info(f"[{transaction_id}] Direct threshold applied: {direct_threshold}, Result: {direct_prediction}")
                
                # Only override if direct_prediction detected fraud but adjusted didn't
                if direct_prediction == 1 and prediction == 0:
                    prediction = 1
                    logging.info(f"[{transaction_id}] Prediction overridden by direct threshold")
            
            logging.info(f"[{transaction_id}] Final prediction: {prediction}, Using threshold: {threshold}")
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logging.error(f"[{transaction_id}] {error_msg}")
            return jsonify({"error": error_msg, "transaction_id": transaction_id}), 500
        
        # Result - IMPORTANT: Keep both original_fraud_probability and fraud_probability for compatibility
        result = {
            'prediction': prediction,
            'prediction_label': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'original_fraud_probability': original_fraud_probability,
            'adjusted_fraud_probability': adjusted_fraud_probability,
            'fraud_probability': original_fraud_probability,  # Keep for backward compatibility
            'adjustment_factor': adjustment_factor,
            'legitimate_probability': legitimate_probability,
            'threshold_used': threshold,
            'threshold_key': threshold_key,
            'transaction_id': transaction_id,
            'feature_count': {
                'original': actual_features,
                'padded_to': expected_features if actual_features < expected_features else actual_features
            }
        }
        
        # Add additional information for the template
        if not request.is_json:
            result['available_thresholds'] = thresholds
            result['threshold_descriptions'] = {
                'gmean': 'Optimal geometric mean of sensitivity and specificity',
                'f1': 'Optimal F1 score (balance of precision and recall)',
                'balanced': 'Standard threshold (0.5)',
                'conservative': 'High Sensitivity (Catch More Fraud, uses 1% threshold)',
                'default': 'Default system threshold'
            }
        
        # Return JSON for API calls
        if request.is_json:
            return jsonify(result)
        
        # Return to template for form submissions
        return render_template('result.html', result=result)
    
    except Exception as e:
        error_msg = str(e)
        logging.error(f"[{transaction_id}] Unhandled error: {error_msg}")
        return jsonify({"error": error_msg, "transaction_id": transaction_id}), 500

# Add a route to see the model's feature scaling stats
@app.route('/model-stats')
def model_stats():
    if scaler is None:
        return jsonify({"error": "Scaler not loaded"}), 500
        
    try:
        # Return useful statistics about the scaler for debugging
        stats = {
            "feature_count": scaler.mean_.shape[0],
            "feature_means": scaler.mean_.tolist(),
            "feature_variances": scaler.var_.tolist(),
            "feature_scales": scaler.scale_.tolist()
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Basic health check endpoint
@app.route('/health')
def health_check():
    health = {
        'status': 'healthy' if ensemble_model is not None else 'degraded',
        'models_loaded': ensemble_model is not None,
        'scaler_loaded': scaler is not None,
        'thresholds_available': list(thresholds.keys()) if thresholds else []
    }
    return jsonify(health)

if __name__ == '__main__':
    app.run(debug=True)