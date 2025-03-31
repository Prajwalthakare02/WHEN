import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=== Deep Dataset Analysis ===")

# Choose the dataset to analyze
datasets = [
    '/workspaces/pajaya/New_Predict/Extended_Realistic_Placement_Dataset.csv',
    '/workspaces/pajaya/New_Predict/Enhanced_Placement_Dataset.csv',
    '/workspaces/pajaya/New_Predict/Guaranteed_Placement_Dataset.csv'
]

for dataset_path in datasets:
    try:
        print(f"\nTesting dataset: {dataset_path}")
        data = pd.read_csv(dataset_path)
        print(f"Dataset shape: {data.shape}")
        
        # Check target distribution
        print("\nTarget distribution:")
        print(data['Placement_Status'].value_counts(normalize=True))
        
        # Extract target variable
        y = data['Placement_Status'].copy()
        
        # Check for trivial model accuracy
        from collections import Counter
        most_common = Counter(y).most_common(1)[0][0]
        trivial_accuracy = sum(1 for val in y if val == most_common) / len(y)
        print(f"Trivial accuracy (always predicting '{most_common}'): {trivial_accuracy:.4f}")
        
        # Simplest possible model - just using CGPA
        if 'CGPA' in data.columns:
            print("\nSimple CGPA threshold test:")
            # Find optimal CGPA threshold
            best_accuracy = 0
            best_threshold = 0
            
            for threshold in np.linspace(data['CGPA'].min(), data['CGPA'].max(), 100):
                predictions = (data['CGPA'] >= threshold).map({True: 'Placed', False: 'Not Placed'})
                acc = sum(predictions == data['Placement_Status']) / len(data)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_threshold = threshold
            
            print(f"Best CGPA threshold: {best_threshold:.2f}, Accuracy: {best_accuracy:.4f}")
            print(f"Better than trivial: {'Yes' if best_accuracy > trivial_accuracy else 'No'}")
        
        # Simplest model with all features
        print("\nBasic model with direct features (no preprocessing):")
        
        # Drop non-feature columns
        X = data.drop(['Placement_Status', 'Student_ID', 'Full_Name', 'Company_Name'], 
                     axis=1, errors='ignore')
        
        # Handle categorical columns directly with label encoding
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, rf_model.predict(X_train))
        test_acc = accuracy_score(y_test, rf_model.predict(X_test))
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Testing accuracy: {test_acc:.4f}")
        print(f"Overfitting: {'Yes' if train_acc - test_acc > 0.1 else 'No'}")
        print(f"Better than random: {'Yes' if test_acc > 0.55 else 'No'}")
        print(f"Better than trivial: {'Yes' if test_acc > trivial_accuracy else 'No'}")
        
        # Feature importance
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 5 important features:")
        print(importances.head(5))
        
        # CRITICAL TEST: Check if target is truly related to features
        print("\nCRITICAL TEST: Target-feature relationship")
        
        # Create a copy of data with shuffled target
        data_shuffled = data.copy()
        data_shuffled['Placement_Status'] = np.random.permutation(data['Placement_Status'])
        
        # Prepare shuffled data
        X_shuffled = data_shuffled.drop(['Placement_Status', 'Student_ID', 'Full_Name', 'Company_Name'], 
                                      axis=1, errors='ignore')
        
        # Handle categorical columns
        for col in X_shuffled.select_dtypes(include=['object']).columns:
            X_shuffled[col] = LabelEncoder().fit_transform(X_shuffled[col])
        
        # Encode target
        y_shuffled = le.fit_transform(data_shuffled['Placement_Status'])
        
        # Split data
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_shuffled, y_shuffled, test_size=0.3, random_state=42, stratify=y_shuffled)
        
        # Train model
        rf_model_s = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_s.fit(X_train_s, y_train_s)
        
        # Evaluate
        shuffled_acc = accuracy_score(y_test_s, rf_model_s.predict(X_test_s))
        
        print(f"Accuracy with shuffled target: {shuffled_acc:.4f}")
        print(f"Original test accuracy: {test_acc:.4f}")
        print(f"Difference: {test_acc - shuffled_acc:.4f}")
        
        if test_acc - shuffled_acc < 0.05:
            print("CRITICAL ISSUE: Target variable appears to be random or unrelated to features!")
            print("This explains why all models get ~50% accuracy.")
        else:
            print("Target variable shows relationship with features. Look for implementation issues.")
        
        print("\n" + "="*40)
        
    except Exception as e:
        print(f"Error processing {dataset_path}: {e}")
