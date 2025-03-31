import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Custom transformer for label encoding (EXACTLY as in notebook cell 2)
class LabelEncoderPipelineFriendly(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.columns_ = None
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = X.columns
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.columns_ is None:
                raise ValueError("Transformer has not been fitted yet and received an array.")
            X = pd.DataFrame(X, columns=self.columns_)
        X_transformed = X.copy()
        for col, le in self.encoders.items():
            X_transformed[col] = le.transform(X_transformed[col].astype(str))
        return X_transformed

def main():
    print("=== Training Placement Prediction Models (Adaptive) ===")
    print("Using adaptive feature selection to work with any dataset structure")
    
    # Define output directory for all models
    output_dir = "/workspaces/pajaya/very_new_predict"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All models will be saved to: {output_dir}")
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    data = pd.read_csv('/workspaces/pajaya/very_new_predict/Ultra_Predictive_Dataset.csv')
    print(f"Dataset loaded successfully: {data.shape}")
    
    # Handle missing values
    data.fillna("Unknown", inplace=True)
    
    # Step 2: Automatically identify numerical and categorical features
    print("\nStep 2: Auto-identifying features...")
    
    # Drop non-feature columns
    drop_cols = ['Student_ID', 'Full_Name', 'Company_Name']
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(col, axis=1)
    
    target = 'Placement_Status'
    
    # Automatically identify numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in numerical_features:
        numerical_features.remove(target)
    
    # Automatically identify categorical features
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if target in categorical_features:
        categorical_features.remove(target)
    
    print(f"Identified {len(numerical_features)} numerical features:")
    print(numerical_features)
    print(f"\nIdentified {len(categorical_features)} categorical features:")
    print(categorical_features)
    
    # Step 3: Create preprocessing pipelines
    print("\nStep 3: Creating preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoderPipelineFriendly())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Step 4: Define target variable and features
    print("\nStep 4: Preparing features and target...")
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Step 5: Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, os.path.join(output_dir, "target_encoder.pkl"))
    print(f"Target variable encoded and saved to {output_dir}/target_encoder.pkl")
    
    # Step 6: Apply SMOTE for class balancing
    print("\nStep 6: Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    # First preprocess data
    X_preprocessed = preprocessor.fit_transform(X)
    # Then apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y_encoded)
    print(f"After SMOTE: X shape: {X_resampled.shape}, y shape: {y_resampled.shape}")
    
    # Step 7: Train-Test split
    print("\nStep 7: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
    print(f"Preprocessor saved to {output_dir}/preprocessor.pkl")
    
    # Step 8: Train Random Forest Model
    print("\nStep 8: Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    
    # Save Random Forest model
    joblib.dump(rf_model, os.path.join(output_dir, "random_forest_model.pkl"))
    print(f"Random Forest model saved to {output_dir}/random_forest_model.pkl")
    
    # Step 9: Train XGBoost Model
    print("\nStep 9: Training XGBoost Model...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.01,
        colsample_bytree=0.8,
        subsample=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_xgb))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))
    
    # Save XGBoost model
    joblib.dump(xgb_model, os.path.join(output_dir, "best_xgb_model.pkl"))
    print(f"XGBoost model saved to {output_dir}/best_xgb_model.pkl")
    
    # Step 10: Train Ensemble Voting Classifier
    print("\nStep 10: Training Ensemble Voting Classifier...")
    voting_model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=500, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)
    
    # Evaluate Voting Classifier
    y_pred_voting = voting_model.predict(X_test)
    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    print(f"Ensemble Model Accuracy: {voting_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_voting))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_voting))
    
    # Save Ensemble model
    joblib.dump(voting_model, os.path.join(output_dir, "ensemble_model.pkl"))
    print(f"Ensemble model saved to {output_dir}/ensemble_model.pkl")
    
    # Step 11: Train Stacking Classifier
    print("\nStep 11: Training Stacking Classifier...")
    estimators = [
        ('lr', LogisticRegression(max_iter=500, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=150, random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=200, 
            max_depth=7, 
            learning_rate=0.01, 
            colsample_bytree=0.8, 
            subsample=1.0, 
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42))
    ]
    
    # Define the final estimator (meta-learner)
    final_estimator = LogisticRegression(max_iter=500, C=0.1, random_state=42)
    
    # Create stacking classifier
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    
    # Train the stacking classifier
    stacking_model.fit(X_train, y_train)
    
    # Evaluate Stacking Classifier
    y_pred_stacking = stacking_model.predict(X_test)
    stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
    print(f"Stacking Classifier Accuracy: {stacking_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_stacking))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_stacking))
    
    # Save Stacking model
    joblib.dump(stacking_model, os.path.join(output_dir, "stacking_classifier_model.pkl"))
    print(f"Stacking Classifier model saved to {output_dir}/stacking_classifier_model.pkl")
    
    # Also save encoders and scaler separately for potential direct use
    joblib.dump({'numerical': numerical_features, 'categorical': categorical_features}, 
               os.path.join(output_dir, "feature_lists.pkl"))
    print(f"Feature lists saved to {output_dir}/feature_lists.pkl")
    
    # Print model comparison
    print("\n=== Model Performance Comparison ===")
    models = {
        "Random Forest": rf_accuracy,
        "XGBoost": xgb_accuracy,
        "Voting Ensemble": voting_accuracy,
        "Stacking Classifier": stacking_accuracy
    }
    
    for model, acc in sorted(models.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {acc:.4f}")
    
    print("\n=== Model Training Complete ===")
    print(f"All models have been saved to: {output_dir}")

if __name__ == "__main__":
    main()
