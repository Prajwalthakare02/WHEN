import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os
from utils import preprocess_data, create_example_input

def train_stacking_model(X_train, y_train):
    """Train the Stacking Classifier model"""
    # Define base estimators
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
    
    # Define the meta-learner
    final_estimator = LogisticRegression(max_iter=500, C=0.1, random_state=42)
    
    # Create and train Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    
    stacking_model.fit(X_train, y_train)
    return stacking_model

def save_model(model, filename="stacking_classifier_model.pkl"):
    """Save the trained model"""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename="stacking_classifier_model.pkl"):
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
        model: Trained Stacking Classifier model
        student_data (DataFrame): Student data
        encoders (dict): Dictionary of label encoders
        scaler: Fitted standard scaler
        
    Returns:
        float: Probability of placement (0-100%)
    """
    from utils import preprocess_data
    
    # Preprocess student data
    processed_data, _, _ = preprocess_data(student_data, is_training=False, 
                                          encoders=encoders, scaler=scaler)
    
    try:
        # Make sure the columns match those used during training
        if hasattr(model, 'feature_names_in_'):
            # Get the features the model was trained on
            model_features = model.feature_names_in_
            
            # Create empty DataFrame with all expected columns
            prediction_data = pd.DataFrame(columns=model_features)
            
            # Fill with zeros - important to avoid NaN values
            prediction_data.loc[0] = 0
            
            # Fill in values from processed data where they match
            common_columns = set(processed_data.columns) & set(model_features)
            for col in common_columns:
                prediction_data[col] = processed_data[col].values
            
            # Make sure there are no NaN values
            prediction_data.fillna(0, inplace=True)
            processed_data = prediction_data
        
        # Also ensure no NaN values in the processed data
        if isinstance(processed_data, pd.DataFrame):
            processed_data.fillna(0, inplace=True)
        
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
        print(f"Error in prediction: {e}")
        # Return a default value in case of error
        return 50.0  # Neutral prediction

def predict_placement(student_data, model=None, encoders=None, scaler=None):
    """
    Predict whether a student will be placed or not
    
    Args:
        student_data (DataFrame): Student data
        model: Trained Stacking model
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

# Example usage
if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC  # Import needed for the stacking classifier
    
    # Create example student data
    example_student = create_example_input()
    
    # Print the student details
    print("Student Details:")
    for col, val in example_student.iloc[0].items():
        print(f"{col}: {val}")
    print("\n")
    
    # Load pre-trained model (assuming it exists)
    model = load_model()
    
    if model is None:
        print("No pre-trained model found. Please train the model first.")
    else:
        # Create dummy encoders and scaler for demonstration
        # In practice, you would load these from saved files
        encoders = {}
        for col in example_student.select_dtypes(include=['object']).columns:
            encoders[col] = LabelEncoder().fit(example_student[col])
        
        scaler = StandardScaler().fit(example_student[['CGPA', 'Soft_Skills_Score', 'Live_Backlogs', 'Cleared_Backlogs']])
            
        # Predict placement
        placement, probability = predict_placement(example_student, model, encoders, scaler)
        
        # Display prediction results
        print(f"Stacking Classifier Model Prediction:")
        print(f"Prediction: {placement}")
        print(f"Placement Probability: {probability:.2f}%")
