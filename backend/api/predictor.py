"""
Placement Prediction module for predicting student placement
"""
import os
import random

# In a real implementation, you would have:
# import joblib
# import numpy as np
# import pandas as pd

def predict_placement(student_data):
    """
    Predict whether a student will be placed based on their academic and skills data
    
    Args:
        student_data (dict): Student's academic and skills data
            - cgpa: float - Student's CGPA (0-10)
            - soft_skills_score: float - Soft skills score (0-10)
            - technical_skills: float - Technical skills score (0-10)
            - leadership_score: float - Leadership score (0-10)
            - experience_years: float - Years of experience
            - live_backlogs: int - Number of active backlogs
            - internships: int - Number of internships completed
            - projects: int - Number of projects completed
            - certifications: int - Number of certifications
            - programming_language: str - Primary programming language
            - branch: str - Branch of study
            - year_of_passing: int - Year of passing
            - gender: str - Gender
    
    Returns:
        tuple: (result, probability)
            - result: "Placed" or "Not Placed"
            - probability: Float between 0 and 100 representing placement probability
    """
    try:
        # In a real implementation, you would:
        # 1. Load your ML model
        # model = joblib.load('path/to/model.pkl')
        
        # 2. Preprocess the input data
        # features = preprocess_data(student_data)
        
        # 3. Make prediction
        # prediction = model.predict_proba(features)
        # result = "Placed" if prediction[0][1] > 0.5 else "Not Placed"
        # probability = prediction[0][1] * 100
        
        # For demonstration, we'll use a simple heuristic algorithm:
        score = 0
        
        # CGPA is a strong indicator (weight: 30%)
        cgpa = float(student_data.get('cgpa', 0))
        score += (cgpa / 10) * 30
        
        # Skills matter (weight: 20%)
        tech_skills = float(student_data.get('technical_skills', 0))
        soft_skills = float(student_data.get('soft_skills_score', 0))
        score += ((tech_skills + soft_skills) / 20) * 20
        
        # Projects and internships (weight: 25%)
        projects = int(student_data.get('projects', 0))
        internships = int(student_data.get('internships', 0))
        score += (min(projects, 5) / 5 * 10) + (min(internships, 3) / 3 * 15)
        
        # Certifications (weight: 10%)
        certifications = int(student_data.get('certifications', 0))
        score += (min(certifications, 5) / 5) * 10
        
        # Backlogs negatively impact (weight: 10%)
        backlogs = int(student_data.get('live_backlogs', 0))
        score -= min(backlogs * 2, 10)
        
        # Experience is valuable (weight: 5%)
        experience = float(student_data.get('experience_years', 0))
        score += (min(experience, 2) / 2) * 5
        
        # Add slight randomness to demonstrate variability
        score += random.uniform(-5, 5)
        
        # Ensure score stays within 0-100
        probability = max(0, min(score, 100))
        
        # Determine placement result
        result = "Placed" if probability > 60 else "Not Placed"
        
        return result, probability
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return a default prediction in case of errors
        return "Not Placed", 45.0

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