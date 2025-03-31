import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
import os

def train_xgboost_model(X_train, y_train):
    """Train the XGBoost model with optimal parameters"""
    # Create and train XGBoost model
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
    return xgb_model

def save_model(model, filename="xgboost_model.pkl"):
    """Save the trained model"""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename="best_xgb_model.pkl"):
    """Load the trained model"""
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        print(f"Model file {filename} not found.")
        return None

def predict_placement_probability(model, student_data, encoders=None, scaler=None):
    """
    Predict placement probability for a student
    
    Args:
        model: Trained XGBoost model
        student_data (DataFrame): Student data
        encoders (dict): Dictionary of label encoders
        scaler: Fitted standard scaler
        
    Returns:
        float: Probability of placement (0-100%)
    """
    from utils import preprocess_data
    
    try:
        # Preprocess student data
        processed_data, _, _ = preprocess_data(student_data, is_training=False, 
                                              encoders=encoders, scaler=scaler)
        
        # Make sure the columns match those used during training
        if hasattr(model, 'feature_names_in_'):
            # Get the features the model was trained on
            model_features = model.feature_names_in_
            
            # Create empty DataFrame with all expected columns
            prediction_data = pd.DataFrame(columns=model_features)
            
            # Fill with zeros
            prediction_data.loc[0] = 0
            
            # Fill in values from processed data where they match
            common_columns = set(processed_data.columns) & set(model_features)
            for col in common_columns:
                prediction_data[col] = processed_data[col].values
            
            # Ensure no NaN values
            prediction_data.fillna(0, inplace=True)
            processed_data = prediction_data
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # Get the probability of positive class (Placed)
            probability = model.predict_proba(processed_data)[0][1] * 100
        else:
            # If the model doesn't have predict_proba, use predict
            prediction = model.predict(processed_data)[0]
            probability = 100 if prediction == 1 else 0
            
        return probability
        
    except Exception as e:
        print(f"Error in XGBoost prediction: {e}")
        return 50.0  # Return neutral prediction in case of error

def predict_placement(student_data, model=None, encoders=None, scaler=None):
    """
    Predict whether a student will be placed or not
    
    Args:
        student_data (DataFrame): Student data
        model: Trained XGBoost model
        encoders (dict): Dictionary of label encoders
        scaler: Fitted standard scaler
        
    Returns:
        tuple: Prediction ("Placed" or "Not Placed"), probability, and top features
    """
    if model is None:
        model = load_model()
        
    # Use utility function to get prediction probability
    probability = predict_placement_probability(model, student_data, encoders, scaler)
    
    # Make binary prediction
    prediction = "Placed" if probability >= 50 else "Not Placed"
    
    # Get feature importances
    feature_importance = None
    
    if hasattr(model, 'feature_importances_'):
        try:
            if hasattr(model, 'feature_names_in_'):
                # The model has stored feature names - use these for reliable matching
                feature_names = model.feature_names_in_
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            else:
                # For models without stored feature names, check lengths before creating DataFrame
                from utils import preprocess_data
                processed_data, _, _ = preprocess_data(student_data, is_training=False, 
                                                    encoders=encoders, scaler=scaler)
                
                # Handle potential length mismatch
                if len(processed_data.columns) == len(model.feature_importances_):
                    feature_importance = pd.DataFrame({
                        'Feature': processed_data.columns,
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                else:
                    # Create feature importance with generic feature names
                    feature_importance = pd.DataFrame({
                        'Feature': [f'Feature_{i}' for i in range(len(model.feature_importances_))],
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    print("Warning: Feature names couldn't be matched with importances.")
        except Exception as e:
            print(f"Warning: Couldn't create feature importance: {e}")
            # Create a placeholder feature importance
            feature_importance = pd.DataFrame({
                'Feature': ['NA'],
                'Importance': [1.0]
            })
    else:
        # Model doesn't have feature importances, create a placeholder
        feature_importance = pd.DataFrame({
            'Feature': ['NA'],
            'Importance': [1.0]
        })
    
    return prediction, probability, feature_importance
