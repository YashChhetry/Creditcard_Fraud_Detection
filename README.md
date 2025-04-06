# Creditcard_Fraud_Detection
Credit Card Fraud Detection System - Packages and Installation Guide
Overview
This repository contains a complete credit card fraud detection system with a machine learning backend and Flask-based web interface. The system uses ensemble learning (XGBoost and Logistic Regression) with SMOTE to handle class imbalance, providing an end-to-end solution for identifying fraudulent credit card transactions.

Required Packages
Python Packages
# Core packages
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
xgboost==1.6.2
imbalanced-learn==0.9.1
Flask==2.0.3

# Visualization
matplotlib==3.5.2
seaborn==0.11.2

# Utilities
pickle-mixin==1.0.2

# Web interface
Werkzeug==2.0.3
Jinja2==3.0.3
Frontend Libraries (included via CDN)
Bootstrap 4.5.2
jQuery 3.5.1
Popper.js 1.16.1
Font Awesome 5.15.1
Installation
Clone this repository:
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:
pip install -r requirements.txt
Prepare your data directory:
mkdir -p "C:\Users\User\Desktop\Credit_Card_Fraud\Data"
# Copy your CSV files into the above directory
Project Structure
credit-card-fraud-detection/
│
├── models/                   # Saved models and scalers
│   ├── smote_ensemble_model.pkl  
│   ├── smote_scaler.pkl
│   └── smote_thresholds.pkl
│
├── templates/                # HTML templates
│   ├── index.html            # Input form
│   └── result.html           # Results display
│
├── static/                   # Static assets
│   └── css/
│
├── model_training.py         # Script to train and save models
├── app.py                    # Flask web application
├── requirements.txt          # Package dependencies
└── README.md                 # This file
Usage
1. Train the Model
python model_training.py
This script will:

Load data from CSV files
Apply SMOTE to balance the classes
Train XGBoost and Logistic Regression models
Create an ensemble model
Find optimal decision thresholds
Save the trained models and thresholds
2. Start the Web Application
python app.py
This will start the Flask application at http://localhost:5000

3. Using the Web Interface
The web interface offers three ways to input data:

Individual Inputs: Enter values for each feature
Batch Input: Paste all features as comma-separated values
Sample Cases: Use pre-defined test cases (legitimate, fraudulent, borderline)
Key Features
Ensemble Learning: Combines XGBoost and Logistic Regression for improved accuracy
SMOTE Balancing: Handles class imbalance typical in fraud detection
Multiple Thresholds: Provides different sensitivity levels for fraud detection
Probability Adjustment: Enhances sensitivity for detecting fraudulent patterns
Detailed Visualization: Clear presentation of fraud probabilities and model confidence
RESTful API: Supports both web interface and API-based prediction
Comprehensive Logging: Tracks all predictions for audit and debugging
Model Performance
The model achieves high accuracy in detecting credit card fraud, with performance metrics as follows:

Accuracy: 99.95%
Precision: 99.98%
Recall: 99.96%
F1 Score: 99.97%
AUC: 99.99%
API Endpoints
GET / - Main web interface
POST /predict - Make a prediction (accepts form data or JSON)
GET /model-stats - View model scaling statistics
GET /health - Check system health status
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The project uses PCA-transformed credit card transaction data for fraud detection
SMOTE implementation is based on the imbalanced-learn library
Web interface built with Flask and Bootstrap
Technical Context
Machine Learning Architecture
This project implements a fraud detection system using an ensemble of two powerful classifiers:

XGBoost: A gradient boosting algorithm that excels at capturing complex patterns in data
Logistic Regression: A simpler linear model that provides good generalization and interpretability
The ensemble combines these models using a soft voting approach, where the final prediction is based on the average predicted probabilities from both models.

SMOTE Implementation
Credit card fraud datasets typically suffer from extreme class imbalance (often <1% fraud). This project addresses this challenge using Synthetic Minority Over-sampling Technique (SMOTE) which:

Creates synthetic examples of the minority class (fraud)
Helps the model learn better patterns for fraud detection
Improves recall without excessively compromising precision
Decision Thresholds
The system provides multiple threshold options:

F1-optimized: Balances precision and recall
G-mean optimized: Balances sensitivity and specificity
Conservative (0.01): High sensitivity for catching more potential fraud
Balanced (0.5): Standard classification threshold
Flask Application Architecture
The Flask app implements a RESTful API and web interface for the fraud detection model:

Uses server-side rendering with Jinja2 templates
Provides comprehensive error handling and logging
Supports both form-based submissions and JSON API calls
Includes a responsive Bootstrap-based UI
Deployment Considerations
For production deployment:

Set
debug=False
in the Flask app
Implement proper authentication for API access
Consider containerization with Docker
Set up monitoring and alerting for model performance
Implement a periodic model retraining pipeline
