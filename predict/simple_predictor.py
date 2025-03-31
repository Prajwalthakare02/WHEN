import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_model(model_name, model_dir="/workspaces/pajaya/New_Predict"):
    """Load a specific model file."""
    model_file = os.path.join(model_dir, f"{model_name}.pkl")
    if os.path.exists(model_file):
        print(f"Loading {model_name}...")
        return joblib.load(model_file)
    else:
        print(f"Model {model_file} not found")
        return None

def create_exact_input():
    """Create example input data with EXACTLY the expected features."""
    # Predefined feature sets based on what was used in training
    # This is specifically designed to match what's in random_forest_model.pkl
    expected_features = [
        'CGPA', 'Soft_Skills_Score', 'Technical_Skills', 'Leadership_Score',
        'Experience_Years', 'Live_Backlogs', 'No_of_Internships', 'Projects',
        'No_of_Certifications', 'Programming_Language', 'Gender', 'Branch', 'Year_of_Passing'
    ]
    
    # Create input data with exactly those features
    data = {feature: [0] for feature in expected_features}
    
    # Set example values
    data['CGPA'] = [8.5]
    data['Soft_Skills_Score'] = [8]
    data['Technical_Skills'] = [9]
    data['Leadership_Score'] = [7]
    data['Experience_Years'] = [1.5]
    data['Live_Backlogs'] = [0]
    data['No_of_Internships'] = [2]
    data['Projects'] = [3]
    data['No_of_Certifications'] = [2]
    data['Programming_Language'] = [0]  # Already encoded as number
    data['Gender'] = [0]  # Already encoded as number
    data['Branch'] = [0]  # Already encoded as number
    data['Year_of_Passing'] = [2023]
    
    return pd.DataFrame(data)

def predict_with_models():
    """Make predictions using all available models."""
    # Create input data with exact structure
    input_data = create_exact_input()
    print(f"Created input data with {len(input_data.columns)} features:")
    print(input_data.columns.tolist())
    
    # Load models
    models = {
        "Random Forest": load_model("random_forest_model"),
        "XGBoost": load_model("best_xgb_model"),
        "Ensemble": load_model("ensemble_model"),
        "Stacking": load_model("stacking_classifier_model")
    }
    
    # Make predictions
    results = {}
    for name, model in models.items():
        if model is not None:
            try:
                print(f"\nMaking prediction with {name} model...")
                
                # Verify correct features
                if hasattr(model, 'feature_names_in_'):
                    expected = set(model.feature_names_in_)
                    actual = set(input_data.columns)
                    
                    missing = expected - actual
                    extra = actual - expected
                    
                    if missing:
                        print(f"Warning: Missing features: {missing}")
                    if extra:
                        print(f"Warning: Extra features: {extra}")
                    
                    # Keep only expected features in right order
                    aligned_input = pd.DataFrame(columns=model.feature_names_in_)
                    aligned_input.loc[0] = 0
                    
                    for col in model.feature_names_in_:
                        if col in input_data.columns:
                            aligned_input[col] = input_data[col].values
                            
                    print(f"Features aligned from {len(input_data.columns)} to {len(aligned_input.columns)}")
                    input_to_use = aligned_input
                else:
                    input_to_use = input_data
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_to_use)
                    if proba.shape[1] >= 2:  # Binary classification
                        probability = proba[0][1] * 100
                    else:
                        probability = proba[0][0] * 100
                else:
                    pred = model.predict(input_to_use)[0]
                    probability = 100 if pred == 1 else 0
                    
                prediction = "Placed" if probability >= 50 else "Not Placed"
                results[name] = {"prediction": prediction, "probability": probability}
                
                print(f"Prediction: {prediction}")
                print(f"Probability: {probability:.2f}%")
                
            except Exception as e:
                print(f"Error with {name} model: {str(e)}")
                results[name] = {"prediction": "Error", "probability": 0}
    
    return results

def main():
    print("=== Simple Student Placement Predictor ===")
    print("Using fixed input data structure to match trained models")
    
    results = predict_with_models()
    
    # Show consensus
    print("\n=== Consensus Prediction ===")
    valid_results = {k: v for k, v in results.items() if v["prediction"] != "Error"}
    
    if valid_results:
        placed_votes = sum(1 for r in valid_results.values() if r["prediction"] == "Placed")
        not_placed_votes = sum(1 for r in valid_results.values() if r["prediction"] == "Not Placed")
        avg_prob = np.mean([r["probability"] for r in valid_results.values()])
        
        print(f"Placed votes: {placed_votes}")
        print(f"Not Placed votes: {not_placed_votes}")
        print(f"Average probability: {avg_prob:.2f}%")
        
        if placed_votes > not_placed_votes:
            print("Overall consensus: PLACED")
        elif not_placed_votes > placed_votes:
            print("Overall consensus: NOT PLACED")
        else:
            print("Overall consensus: TIED - UNCERTAIN")
    else:
        print("No valid predictions were made.")

if __name__ == "__main__":
    main()
