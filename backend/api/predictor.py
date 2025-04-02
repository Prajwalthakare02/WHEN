"""
Placement Prediction module for predicting student placement
"""
import os
import random
import pandas as pd

# In a real implementation, you would have:
# import joblib
# import numpy as np
# import pandas as pd

def predict_placement(student_data):
    """
    Predict whether a student will be placed based on their academic data
    
    Args:
        student_data (dict): Student's academic data
            - ssc_percentage: float - 10th standard percentage
            - hsc_percentage: float - 12th standard percentage
            - degree_percentage: float - Degree percentage 
            - etest_percentage: float - Employability test percentage
            - mba_percentage: float - MBA percentage
            - work_experience: int - Months of work experience
            - gender: str - Gender (M/F)
            - specialisation: str - Specialisation (Mkt&Fin, Mkt&HR)
    
    Returns:
        bool: True if placed, False if not placed
    """
    try:
        # Convert data to appropriate types
        ssc_percentage = float(student_data.get('ssc_percentage', 0))
        hsc_percentage = float(student_data.get('hsc_percentage', 0))
        degree_percentage = float(student_data.get('degree_percentage', 0))
        etest_percentage = float(student_data.get('etest_percentage', 0))
        mba_percentage = float(student_data.get('mba_percentage', 0))
        work_experience = int(student_data.get('work_experience', 0))
        gender = student_data.get('gender', 'M')
        specialisation = student_data.get('specialisation', 'Mkt&Fin')
        
        # Convert to dataframe for consistency with real ML implementation
        df = pd.DataFrame({
            'ssc_p': [ssc_percentage],
            'hsc_p': [hsc_percentage],
            'degree_p': [degree_percentage],
            'etest_p': [etest_percentage],
            'mba_p': [mba_percentage],
            'workex': [1 if work_experience > 0 else 0],
            'gender_M': [1 if gender == 'M' else 0],
            'gender_F': [1 if gender == 'F' else 0],
            'specialisation_Mkt&Fin': [1 if specialisation == 'Mkt&Fin' else 0],
            'specialisation_Mkt&HR': [1 if specialisation == 'Mkt&HR' else 0]
        })
        
        # For demonstration, use a simple rule-based model
        score = 0
        
        # Academic performance (60% weight)
        score += (ssc_percentage / 100) * 10    # 10% weight
        score += (hsc_percentage / 100) * 10    # 10% weight
        score += (degree_percentage / 100) * 15  # 15% weight
        score += (mba_percentage / 100) * 25     # 25% weight
        
        # Employability test (20% weight)
        score += (etest_percentage / 100) * 20
        
        # Work experience (10% weight)
        score += min(1, work_experience / 12) * 10
        
        # Specialization bonus (5% weight)
        if specialisation == 'Mkt&Fin':
            score += 5  # Finance specialization has slightly better placement rates
        else:
            score += 3  # HR specialization
            
        # Gender factor (5% weight)
        # This is purely for demonstration and should not be used in real models
        # Real models should be bias-free
        if gender == 'M':
            score += 3
        else:
            score += 5
            
        # Add slight randomness (±5%)
        score += random.uniform(-5, 5)
        
        # Determine placement result (threshold: 60%)
        is_placed = score >= 60
        
        return is_placed
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Default to not placed if there's an error
        return False

def preprocess_data(student_data):
    """
    Preprocess student data for model input
    In a real implementation, this would convert categorical variables,
    normalize numerical features, etc.
    
    Args:
        student_data (dict): Raw student data
    
    Returns:
        processed_data: Processed data ready for model input
    """
    # This is a placeholder function
    # In a real implementation, you would:
    # 1. Convert categorical variables to numerical (one-hot encoding)
    # 2. Normalize numerical features
    # 3. Handle missing values
    # 4. Create feature vector in the format expected by your model
    
    return student_data 