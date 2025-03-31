import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def train_random_forest_model(X_train, y_train):
    """Train the Random Forest model"""
    # Create and train Random Forest model with optimal parameters
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    # Store feature names used during training
    if hasattr(X_train, 'columns'):
        rf_model.feature_names_in_ = X_train.columns.tolist()
    return rf_model

def save_model(model, filename="random_forest_model.pkl"):
    """Save the trained model"""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename="random_forest_model.pkl"):
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
        model: Trained Random Forest model
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
        print(f"Error in Random Forest prediction: {e}")
        return 50.0  # Return neutral prediction in case of error

def predict_placement(student_data, model=None, encoders=None, scaler=None):
    """
    Predict whether a student will be placed or not
    
    Args:
        student_data (DataFrame): Student data
        model: Trained Random Forest model
        encoders (dict): Dictionary of label encoders
        scaler: Fitted standard scaler
        
    Returns:
        tuple: Prediction ("Placed" or "Not Placed") and probability
    """
    if model is None:
        model = load_model()
        
    # Use utility function to get prediction probability
    probability = predict_placement_probability(model, student_data, encoders, scaler)
    
    # Make binary prediction
    prediction = "Placed" if probability >= 50 else "Not Placed"
    
    return prediction, probability
