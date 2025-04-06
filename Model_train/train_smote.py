import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc)
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Step 1: Load data function
def load_data(file_paths):
    file_paths = [
        r'C:\Users\User\Desktop\Credit_Card_Fraud\Data\creditcard_2023.csv',
        r'C:\Users\User\Desktop\Credit_Card_Fraud\Data\creditcard.csv',
        r'C:\Users\User\Desktop\Credit_Card_Fraud\Data\creditcard1.csv'
    ]
    dataframes = []
    for file in file_paths:
        if os.path.exists(file):
            print(f"Loading file: {file}")
            df = pd.read_csv(file)
            print(f"Loaded data shape: {df.shape}")
            dataframes.append(df)
        else:
            print(f"Warning: File {file} does not exist: {file}")
    
    if not dataframes:
        raise FileNotFoundError(f"No CSV files found in the provided paths")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    return combined_df

# Step 2: Preprocess data with SMOTE
def preprocess_data_with_smote(df, smote_sampling_strategy=0.5, random_state=42):
    # Check if 'Class' column exists
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
    
    # Check class imbalance before SMOTE
    print("Original class distribution:")
    class_counts = Counter(y)
    print(class_counts)
    
    # Calculate fraud percentage
    fraud_pct = class_counts[1] / sum(class_counts.values()) * 100
    print(f"Original fraud percentage: {fraud_pct:.2f}%")
    
    # Visualize original class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Original Class Distribution')
    plt.yscale('log')  # Log scale for better visualization
    plt.savefig('original_class_distribution.png')
    plt.close()
    
    # Split data before applying SMOTE (to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Scale the features before SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE only to the training data
    print("\nApplying SMOTE...")
    smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Check class distribution after SMOTE
    print("Class distribution after SMOTE:")
    resampled_class_counts = Counter(y_train_resampled)
    print(resampled_class_counts)
    
    # Calculate new fraud percentage
    resampled_fraud_pct = resampled_class_counts[1] / sum(resampled_class_counts.values()) * 100
    print(f"Resampled fraud percentage: {resampled_fraud_pct:.2f}%")
    
    # Visualize resampled class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_train_resampled)
    plt.title('Class Distribution After SMOTE')
    plt.savefig('smote_class_distribution.png')
    plt.close()
    
    # Compare original vs resampled distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train)
    plt.title('Original Training Data')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_train_resampled)
    plt.title('SMOTE Resampled Training Data')
    
    plt.tight_layout()
    plt.savefig('original_vs_smote.png')
    plt.close()
    
    print(f"\nOriginal training set shape: {X_train_scaled.shape}")
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    
    return X, y, X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler

# Step 3: Visualize SMOTE effect
def visualize_smote_effect(X_train, y_train, X_resampled, y_resampled, n_features=5):
    # Convert to DataFrames for easier handling
    cols = [f'V{i+1}' for i in range(X_train.shape[1])]
    df_original = pd.DataFrame(X_train, columns=cols)
    df_original['Class'] = y_train
    
    df_resampled = pd.DataFrame(X_resampled, columns=cols)
    df_resampled['Class'] = y_resampled
    
    # Select top features to visualize
    top_features = cols[:n_features]
    
    # Create visualization grid
    fig, axes = plt.subplots(n_features, 2, figsize=(15, 4*n_features))
    
    for i, feature in enumerate(top_features):
        # Original distribution
        sns.kdeplot(
            data=df_original, x=feature, hue='Class', 
            ax=axes[i, 0], palette=['blue', 'red']
        )
        axes[i, 0].set_title(f'Original: {feature} Distribution')
        axes[i, 0].legend(['Legitimate', 'Fraud'])
        
        # SMOTE distribution
        sns.kdeplot(
            data=df_resampled, x=feature, hue='Class', 
            ax=axes[i, 1], palette=['blue', 'red']
        )
        axes[i, 1].set_title(f'After SMOTE: {feature} Distribution')
        axes[i, 1].legend(['Legitimate', 'Fraud'])
    
    plt.tight_layout()
    plt.savefig('smote_feature_distributions.png')
    plt.close()

# Step 4: Train models
def train_ensemble_with_smote(X_train_resampled, y_train_resampled, X_test, y_test):
    # Initialize models - with adjusted parameters for balanced data
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
        # No need for scale_pos_weight with balanced data
    )
    
    lr_model = LogisticRegression(
        C=1.0,
        # No need for class_weight with balanced data
        random_state=42,
        max_iter=1000,
        solver='liblinear'
    )
    
    # Train individual models
    print("Training XGBoost model on SMOTE-balanced data...")
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    print("Training Logistic Regression model on SMOTE-balanced data...")
    lr_model.fit(X_train_resampled, y_train_resampled)
    
    # Create ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    
    print("Training ensemble model...")
    ensemble_model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate models
    models = {
        'XGBoost': xgb_model,
        'Logistic Regression': lr_model,
        'Ensemble': ensemble_model
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{name} Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # More detailed metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'smote_{name.lower().replace(" ", "_")}_cm.png')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'smote_{name.lower().replace(" ", "_")}_roc.png')
        plt.close()
    
    return ensemble_model, models

# Step 5: Find optimal threshold
def find_optimal_threshold(model, X_val, y_val):
    # Get predicted probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_val, y_proba)
    
    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # Find the optimal threshold for g-mean
    ix = np.argmax(gmeans)
    gmean_threshold = thresholds_roc[ix]
    
    # Calculate precision and recall for different thresholds
    precision_values = []
    recall_values = []
    f1_values = []
    threshold_values = np.arange(0.1, 1.0, 0.05)
    
    for threshold in threshold_values:
        y_pred = (y_proba >= threshold).astype(int)
        precision_values.append(precision_score(y_val, y_pred))
        recall_values.append(recall_score(y_val, y_pred))
        f1_values.append(f1_score(y_val, y_pred))
    
    # Find optimal threshold for F1 score
    f1_threshold = threshold_values[np.argmax(f1_values)]
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_values, precision_values, label='Precision')
    plt.plot(threshold_values, recall_values, label='Recall')
    plt.plot(threshold_values, f1_values, label='F1 Score')
    plt.axvline(x=f1_threshold, color='r', linestyle='--', label=f'Optimal F1 Threshold: {f1_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall and F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('smote_threshold_optimization.png')
    plt.close()
    
    # Collect optimal thresholds
    thresholds = {
        'gmean': gmean_threshold,
        'f1': f1_threshold,
        'balanced': 0.5,  # Standard threshold for balanced data
        'conservative': 0.3  # Lower threshold to catch more fraud
    }
    
    # Print threshold information
    print("\nOptimal Thresholds:")
    for name, threshold in thresholds.items():
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"{name.capitalize()} Threshold: {threshold:.3f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    return thresholds

# Step 6: Full pipeline
def credit_card_fraud_detection_with_smote():
    # 1. Load data
    data_dir = "C:\\Users\\User\\Desktop\\Credit_Card_Fraud\\Data"
    file_paths = [
        os.path.join(data_dir, "creditcard_2023.csv"),
        os.path.join(data_dir, "creditcard.csv"),
        os.path.join(data_dir, "creditcard1.csv")
    ]
    
    print("Loading data...")
    df = load_data(file_paths)
    
    # 2. Preprocess data with SMOTE (use 0.5 for 1:2 ratio of fraud:normal)
    print("\nPreprocessing data with SMOTE...")
    X, y, X_train_resampled, X_test, y_train_resampled, y_test, scaler = preprocess_data_with_smote(
        df, smote_sampling_strategy=0.5
    )
    
    # 3. Visualize the effect of SMOTE on feature distributions
    print("\nVisualizing SMOTE effect...")
    # Convert indices to integers for indexing
    train_indices = y_train_resampled.index.tolist() if hasattr(y_train_resampled, 'index') else range(len(y_train_resampled))
    
    # Get sample of original training data for visualization
    X_train_sample = X.iloc[train_indices[:1000]] if hasattr(X, 'iloc') else X[train_indices[:1000]]
    y_train_sample = y.iloc[train_indices[:1000]] if hasattr(y, 'iloc') else y[train_indices[:1000]]
    
    # Apply the same transformation
    X_train_sample_scaled = scaler.transform(X_train_sample)
    
    # Use a sample of the resampled data for visualization
    X_train_resampled_sample = X_train_resampled[:1000]
    y_train_resampled_sample = y_train_resampled[:1000]
    
    visualize_smote_effect(X_train_sample_scaled, y_train_sample, 
                          X_train_resampled_sample, y_train_resampled_sample)
    
    # 4. Train models on SMOTE-balanced data
    print("\nTraining models on SMOTE-balanced data...")
    ensemble_model, individual_models = train_ensemble_with_smote(
        X_train_resampled, y_train_resampled, X_test, y_test
    )
    
    # 5. Find optimal threshold (using a portion of the test set as validation)
    # Split test set to create a validation set
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    print("\nFinding optimal classification threshold...")
    thresholds = find_optimal_threshold(ensemble_model, X_val, y_val)
    
    # 6. Evaluate final performance with optimal threshold
    print("\nFinal evaluation with optimal threshold...")
    y_proba = ensemble_model.predict_proba(X_test_final)[:, 1]
    y_pred = (y_proba >= thresholds['f1']).astype(int)
    
    print("\nFinal model performance with optimal threshold:")
    print(f"Accuracy: {accuracy_score(y_test_final, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test_final, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test_final, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test_final, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_final, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test_final, y_pred))
    
    # 7. Save models and threshold
    print("\nSaving models and threshold...")
    os.makedirs('models', exist_ok=True)
    
    # Save ensemble model
    with open('models/smote_ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    # Save individual models
    for name, model in individual_models.items():
        filename = name.lower().replace(' ', '_')
        with open(f'models/smote_{filename}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Save scaler
    with open('models/smote_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save optimal thresholds
    with open('models/smote_thresholds.pkl', 'wb') as f:
        pickle.dump(thresholds, f)
    
    print("Models and thresholds saved to 'models' directory")
    
    return ensemble_model, individual_models, scaler, thresholds

# Run the complete pipeline
if __name__ == "__main__":
    ensemble_model, individual_models, scaler, thresholds = credit_card_fraud_detection_with_smote()