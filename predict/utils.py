import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_data(filepath="Extended_Realistic_Placement_Dataset.csv"):
    """Load the dataset and return basic statistics"""
    data = pd.read_csv(filepath)
    return data

def preprocess_for_training(data, verbose=True):
    """Process data for training, exactly matching the notebook implementation"""
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    df.fillna("Unknown", inplace=True)
    
    # Print target distribution if verbose
    if verbose:
        print("\nTarget distribution:")
        print(df['Placement_Status'].value_counts())
        print(f"Class balance: {df['Placement_Status'].value_counts(normalize=True)}")
    
    # Drop unnecessary columns
    if 'Student_ID' in df.columns:
        df = df.drop('Student_ID', axis=1)
    if 'Full_Name' in df.columns:
        df = df.drop('Full_Name', axis=1)
    if 'Company_Name' in df.columns:
        df = df.drop('Company_Name', axis=1)
    
    # Handle categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('Placement_Status')  # Remove target from encoding
    
    # Create encoders dictionary
    encoders = {}
    
    # Encode categorical features
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df['Placement_Status'] = target_encoder.fit_transform(df['Placement_Status'])
    
    # Scale numerical features
    numerical_features = ['CGPA', 'Soft_Skills_Score', 'Live_Backlogs', 'Cleared_Backlogs', 'No_of_Internships', 'Projects']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Define X and y
    X = df.drop('Placement_Status', axis=1)
    y = df['Placement_Status']
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    if verbose:
        print(f"After SMOTE: X shape: {X_resampled.shape}, y shape: {y_resampled.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(y_resampled)}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    return X_train, X_test, y_train, y_test, encoders, scaler, target_encoder

def preprocess_data(data, is_training=False, encoders=None, scaler=None):
    """Process data for training or prediction"""
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    df.fillna("Unknown", inplace=True)
    
    # Drop unnecessary columns if they exist
    if 'Student_ID' in df.columns:
        df = df.drop('Student_ID', axis=1)
    if 'Full_Name' in df.columns:
        df = df.drop('Full_Name', axis=1)
    if 'Company_Name' in df.columns and is_training:
        df = df.drop('Company_Name', axis=1)
    
    # Initialize encoders dictionary if we're training
    if is_training:
        encoders = {}
    
    # Encode categorical variables (except target if training)
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Placement_Status' or not is_training:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else:
                # If this column doesn't have an encoder yet, create one
                if encoders is None or col not in encoders:
                    le = LabelEncoder()
                    le.fit(df[col])
                    if encoders is None:
                        encoders = {}
                    encoders[col] = le
                else:
                    # Use existing encoder
                    le = encoders[col]
                    
                # Handle unseen categories
                df[col] = df[col].map(lambda x: 'Unknown' if x not in le.classes_ else x)
                try:
                    df[col] = le.transform(df[col])
                except:
                    # Fallback for any transformation errors
                    df[col] = 0
    
    # Scale numerical features
    numerical_features = ['CGPA', 'Soft_Skills_Score', 'Live_Backlogs', 'Cleared_Backlogs', 'No_of_Internships', 'Projects']
    numerical_features = [col for col in numerical_features if col in df.columns]
    
    # Only scale if we have numerical features
    if len(numerical_features) > 0:
        if is_training:
            scaler = StandardScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
        else:
            if scaler is None:
                # Create a new scaler if none exists
                scaler = StandardScaler()
                scaler.fit(df[numerical_features])
            try:
                df[numerical_features] = scaler.transform(df[numerical_features])
            except:
                # Fallback in case of any errors
                for col in numerical_features:
                    df[col] = (df[col] - df[col].mean()) / (df[col].std() if df[col].std() != 0 else 1)
    
    # Ensure there are no NaN values in the final dataset
    df.fillna(0, inplace=True)
    
    return df, encoders, scaler

def predict_placement_probability(model, input_data, encoders, scaler):
    """
    Predict the placement probability using the given model.
    
    Args:
        model: Trained model
        input_data (DataFrame): Input data for prediction
        encoders (dict): Dictionary of label encoders
        scaler (StandardScaler): Fitted standard scaler
        
    Returns:
        float: Probability of placement
    """
    # Preprocess the input data
    processed_data, _, _ = preprocess_data(input_data, is_training=False, 
                                         encoders=encoders, scaler=scaler)
    
    # Handle feature name mismatch by checking if model has feature_names_in_ attribute
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        prediction_data = pd.DataFrame(columns=model_features)
        prediction_data.loc[0] = 0  # Initialize with zeros
        
        # Copy data from processed_data where column names match
        for col in processed_data.columns:
            if col in model_features:
                prediction_data[col] = processed_data[col].values
        
        # Ensure no NaN values
        prediction_data.fillna(0, inplace=True)
        processed_data = prediction_data
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(processed_data)[0][1] * 100
    else:
        prediction = model.predict(processed_data)[0]
        probability = 100 if prediction == 1 else 0
        
    return probability

def create_example_input():
    """Create an example input for model prediction"""
    example_data = {
        'Student_ID': ['S1234'],
        'Full_Name': ['Example Student'],
        'CGPA': [8.5],
        'Skills': ['Python, SQL'],
        'Certifications': ['AWS Certified'],
        'Internship_Experience': ['1 year'],
        'Soft_Skills_Score': [8],
        'Live_Backlogs': [0],
        'Cleared_Backlogs': [1],
        'Technical_Projects': ['Data Science'],
        'Workshops_Attended': ['AI Summit'],
        'Student_Club_Participation': ['CSI'],
        'Awards_Achievements': ['Hackathon Winner'],
        'Programming_Languages': ['Python, Java'],
        'Gender': ['Male'],
        'Year_of_Passing': [2023],
        'Branch': ['Computer Science'],
        'Internships': ['Data Science Intern'],
        'No_of_Internships': [2],
        'Projects': [3],
        'Domain_of_Interest': ['Data Science']
    }
    return pd.DataFrame(example_data)

def create_synthetic_test_data(n_samples=100, seed=42):
    """Create synthetic data for testing the models"""
    np.random.seed(seed)
    
    # Creating synthetic features with stronger signal
    cgpa = np.random.normal(7.5, 1.5, n_samples)  # Higher CGPA
    skills_score = np.random.normal(7, 2, n_samples)  # Higher skills scores
    backlogs_live = np.random.randint(0, 3, n_samples)  # Fewer backlogs
    backlogs_cleared = np.random.randint(0, 4, n_samples)
    internships = np.random.randint(1, 4, n_samples)  # More internships
    projects = np.random.randint(2, 6, n_samples)  # More projects
    
    # Create target with strong correlation to features
    placement_prob = (cgpa/10 + skills_score/10 + internships/3 + projects/5 - backlogs_live/3)/3
    placement = (placement_prob > 0.55).astype(int)  # Creates ~70% placement rate
    
    # Create dummy categorical features
    gender = np.random.choice(['Male', 'Female'], n_samples)
    branch = np.random.choice(['CS', 'IT', 'ECE', 'ME'], n_samples)
    
    # Build DataFrame
    data = pd.DataFrame({
        'CGPA': cgpa,
        'Soft_Skills_Score': skills_score,
        'Live_Backlogs': backlogs_live,
        'Cleared_Backlogs': backlogs_cleared,
        'No_of_Internships': internships,
        'Projects': projects,
        'Gender': gender,
        'Branch': branch,
        'Placement_Status': placement
    })
    
    return data
