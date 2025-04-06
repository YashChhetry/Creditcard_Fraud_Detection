import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and preprocess the data
def load_data(file_paths):
    """
    Load multiple CSV files and concatenate them
    
    Args:
        file_paths: List of paths to CSV files
    
    Returns:
        Combined DataFrame
    """
    dataframes = []
    for file in file_paths:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def preprocess_data(df):
    """
    Preprocess the data for fraud detection
    
    Args:
        df: Input DataFrame with PCA-transformed credit card data
    
    Returns:
        X: Features
        y: Target variable
        X_train, X_test, y_train, y_test: Train-test split data
    """
    # Check if 'Class' column exists (typical label for fraud detection)
    # If not, assume the last column is the target
    if 'Class' in df.columns:
        target_col = 'Class'
    else:
        target_col = df.columns[-1]
    
    print(f"Using '{target_col}' as the target column")
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Since the data is already PCA-transformed, scaling may not be necessary
    # but we'll include it as a best practice
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check class imbalance
    print("Class distribution:")
    print(y.value_counts())
    print(f"Fraud percentage: {y.mean() * 100:.2f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X, y, X_train, X_test, y_train, y_test, scaler

# Step 2: Train models and create ensemble
def train_models(X_train, y_train, X_test, y_test):
    """
    Train XGBoost and Logistic Regression models and combine them using ensemble learning
    
    Args:
        X_train, y_train, X_test, y_test: Train-test split data
    
    Returns:
        ensemble_model: Trained ensemble model
        individual_models: Dictionary of trained individual models
    """
    # Initialize models
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    lr_model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='liblinear'
    )
    
    # Train individual models
    print("Training XGBoost model...")
    xgb_model.fit(X_train, y_train)
    
    print("Training Logistic Regression model...")
    lr_model.fit(X_train, y_train)
    
    # Create ensemble model (voting classifier)
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lr', lr_model)
        ],
        voting='soft'  # Use probabilities for voting
    )
    
    print("Training ensemble model...")
    ensemble_model.fit(X_train, y_train)
    
    # Evaluate models
    models = {
        'XGBoost': xgb_model,
        'Logistic Regression': lr_model,
        'Ensemble': ensemble_model
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
    
    return ensemble_model, models

# Step 3: Save models for use in Flask
def save_models(ensemble_model, individual_models, scaler, output_dir='models'):
    """
    Save trained models and scaler for later use in Flask
    
    Args:
        ensemble_model: Trained ensemble model
        individual_models: Dictionary of trained individual models
        scaler: Trained StandardScaler
        output_dir: Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ensemble model
    with open(f"{output_dir}/ensemble_model.pkl", 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    # Save individual models
    for name, model in individual_models.items():
        filename = name.lower().replace(' ', '_')
        with open(f"{output_dir}/{filename}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
    
    # Save scaler
    with open(f"{output_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Models and scaler saved to '{output_dir}' directory")

# Step 4: Generate visualizations to understand model performance
def generate_visualizations(models, X_test, y_test, output_dir='visualizations'):
    """
    Generate visualizations to understand model performance
    
    Args:
        models: Dictionary of trained models
        X_test, y_test: Test data
        output_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate confusion matrices
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{output_dir}/{name.lower().replace(' ', '_')}_cm.png")
        plt.close()
    
    # For XGBoost, we can visualize feature importance
    if 'XGBoost' in models:
        xgb_model = models['XGBoost']
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(xgb_model, max_num_features=20)
        plt.title('XGBoost Feature Importance')
        plt.savefig(f"{output_dir}/xgboost_feature_importance.png")
        plt.close()
    
    print(f"Visualizations saved to '{output_dir}' directory")

# Main function to run the entire pipeline
def main():
    # File paths - replace with your actual file paths
    file_paths = [
        r'C:\Users\User\Desktop\Credit_Card_Fraud\Data\creditcard_2023.csv',
        r'C:\Users\User\Desktop\Credit_Card_Fraud\Data\creditcard.csv',
        r'C:\Users\User\Desktop\Credit_Card_Fraud\Data\creditcard1.csv'
    ]
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(file_paths)
    X, y, X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train models
    print("Training models...")
    ensemble_model, individual_models = train_models(X_train, y_train, X_test, y_test)
    
    # Save models
    print("Saving models...")
    save_models(ensemble_model, individual_models, scaler)
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(individual_models, X_test, y_test)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()