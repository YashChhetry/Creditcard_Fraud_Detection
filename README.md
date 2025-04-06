
# 💳 Credit Card Fraud Detection System

## 📄 Overview

This repository contains a complete **credit card fraud detection system** with a machine learning backend and Flask-based web interface. The system uses **ensemble learning** (XGBoost + Logistic Regression) with **SMOTE** to handle class imbalance, providing an end-to-end solution for identifying fraudulent credit card transactions.

---

## 📦 Required Packages

### 🐍 Python Packages

```txt
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
```

### 🌐 Frontend Libraries (via CDN)

- Bootstrap 4.5.2  
- jQuery 3.5.1  
- Popper.js 1.16.1  
- Font Awesome 5.15.1  

---

## ⚙️ Installation

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2️⃣ Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install the required packages:

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare your data directory:

```bash
mkdir -p "C:\Users\User\Desktop\Credit_Card_Fraud\Data"
```

> 📂 Copy your CSV files into the above directory

---

## 🧱 Project Structure

```
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
```

---

## 🚀 Usage

### 🧠 1. Train the Model

```bash
python model_training.py
```

This script will:

- Load data from CSV files  
- Apply **SMOTE** to balance the classes  
- Train **XGBoost** and **Logistic Regression** models  
- Create an **ensemble model**  
- Find optimal **decision thresholds**  
- Save the trained models and thresholds  

---

### 🌐 2. Start the Web Application

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

### 💻 3. Using the Web Interface

The web interface offers three input options:

- **Individual Inputs:** Enter values for each feature  
- **Batch Input:** Paste all features as comma-separated values  
- **Sample Cases:** Use pre-defined test cases  
  - Legitimate  
  - Fraudulent  
  - Borderline  

---

## ⭐ Key Features

- 🔁 **Ensemble Learning**: Combines XGBoost and Logistic Regression  
- ⚖️ **SMOTE Balancing**: Handles class imbalance  
- 🎯 **Multiple Thresholds**: Offers varied sensitivity levels  
- 📊 **Probability Adjustment**: Enhances fraud detection sensitivity  
- 📈 **Detailed Visualization**: Shows fraud probability and confidence  
- 🔗 **RESTful API**: Supports both UI and API calls  
- 🧾 **Comprehensive Logging**: Tracks predictions for auditing/debugging  

---

## 📈 Model Performance

| Metric     | Score   |
|------------|---------|
| Accuracy   | 99.95%  |
| Precision  | 99.98%  |
| Recall     | 99.96%  |
| F1 Score   | 99.97%  |
| AUC        | 99.99%  |

---

## 🔌 API Endpoints

| Method | Endpoint         | Description                            |
|--------|------------------|----------------------------------------|
| GET    | `/`              | Main web interface                     |
| POST   | `/predict`       | Make a prediction (form or JSON)       |
| GET    | `/model-stats`   | View model scaling statistics          |
| GET    | `/health`        | Check system health                    |

---

## 🔍 Technical Context

### 🧠 Machine Learning Architecture

- **XGBoost**: Gradient boosting for complex pattern recognition  
- **Logistic Regression**: Simple, interpretable linear model  
- **Soft Voting Ensemble**: Combines predictions via averaged probabilities  

### 🧬 SMOTE Implementation

- Tackles extreme class imbalance (<1% fraud)  
- Creates synthetic minority class samples  
- Improves recall while maintaining precision  

### 🎚️ Decision Thresholds

| Type         | Description                              |
|--------------|------------------------------------------|
| F1-optimized | Balances precision and recall            |
| G-mean       | Balances sensitivity and specificity     |
| 0.01         | Conservative (high sensitivity)          |
| 0.5          | Balanced (standard classification)       |

---

## 🏗️ Flask Application Architecture

- 🧱 Server-side rendering with **Jinja2**  
- ✅ Error handling and logging  
- 🧾 Form and JSON API support  
- 🎨 Responsive UI with **Bootstrap**

---

## 🚀 Deployment Considerations

- Set `debug=False` for production  
- Add authentication to secure the API  
- Containerize with **Docker**  
- Add **monitoring & alerting**  
- Set up **periodic model retraining**

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PCA-transformed credit card transaction data  
- SMOTE via `imbalanced-learn`  
- Flask + Bootstrap web interface  

---

Let me know if you want a badge section (build status, license, Python version), Dockerfile instructions, or contribution guidelines added!
