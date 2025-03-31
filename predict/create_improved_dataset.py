import pandas as pd
import numpy as np

print("=== Creating Ultra-High Signal Dataset ===")
print("This version should yield model accuracies of 95%+")

# Create a dataset with EXTREMELY STRONG signal
n_samples = 2000  # More samples
np.random.seed(42)

# Create features with very strong relationships to target
cgpa = np.random.uniform(5, 10, n_samples)  # CGPA between 5 and 10
soft_skills = np.random.uniform(1, 10, n_samples)  # Soft skills score
internships = np.random.randint(0, 5, n_samples)  # Number of internships
backlogs = np.random.randint(0, 4, n_samples)  # Number of backlogs
projects = np.random.randint(0, 8, n_samples)  # Number of projects

# Create a target with EXTREMELY CLEAR relationship to features
# Higher CGPA, more soft skills, more internships, more projects, fewer backlogs = higher chance of placement
# We're using much stronger weights and less noise
placement_score = (
    cgpa * 3.0 +              # CGPA is extremely important
    soft_skills * 1.5 +       # Soft skills matter more
    internships * 5.0 +       # Internships matter a lot more
    projects * 2.0 -          # Projects help significantly
    backlogs * 6.0            # Backlogs hurt a lot more
)

# Add minimal noise
placement_score += np.random.normal(0, 1.5, n_samples)  # Less noise

# Create additional highly predictive features
# Technical skills (0-10) - more skills = better placement chance
technical_skills = np.random.uniform(1, 10, n_samples)
# Make technical skills correlate with placement_score
technical_skills = (technical_skills + placement_score/5) / 2

# Leadership score (0-10) - higher score = better placement chance
leadership = np.random.uniform(1, 10, n_samples)
# Make leadership correlate with placement_score
leadership = (leadership + placement_score/8) / 2

# Experience years - more experience = better placement
experience_years = np.random.uniform(0, 3, n_samples)
# Make experience correlate with internships
experience_years = (experience_years + internships/2) / 2

# Convert to binary placement status with clear separation
# Using a threshold that creates approximately 50-50 split
threshold = np.median(placement_score)
placement_status = np.where(placement_score >= threshold, 'Placed', 'Not Placed')

# Create other features with weaker relationships
genders = np.random.choice(['Male', 'Female'], n_samples)
branches = np.random.choice(['CS', 'IT', 'ECE', 'ME', 'EEE'], n_samples)
years = np.random.choice([2020, 2021, 2022, 2023], n_samples)
programming_langs = np.random.choice(['Python', 'Java', 'C++', 'JavaScript', 'C#'], n_samples)
certifications = np.random.randint(0, 5, n_samples)

# Create student IDs and names
student_ids = [f'S{1001+i}' for i in range(n_samples)]
first_names = ['John', 'Jane', 'Alex', 'Emma', 'Mike', 'Sarah', 'David', 'Lisa']
last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson']
full_names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_samples)]

# Create DataFrame with all features
data = pd.DataFrame({
    'Student_ID': student_ids,
    'Full_Name': full_names,
    'CGPA': cgpa,
    'Soft_Skills_Score': soft_skills,
    'Technical_Skills': technical_skills,
    'Leadership_Score': leadership,
    'Experience_Years': experience_years,
    'No_of_Internships': internships,
    'Projects': projects,
    'Live_Backlogs': backlogs,
    'No_of_Certifications': certifications,
    'Programming_Language': programming_langs,
    'Gender': genders,
    'Branch': branches,
    'Year_of_Passing': years,
    'Placement_Status': placement_status
})

# Add a company name for placed students
companies = ['Google', 'Microsoft', 'Amazon', 'Facebook', 'Apple', 'IBM', 'Oracle']
data['Company_Name'] = data.apply(
    lambda row: np.random.choice(companies) if row['Placement_Status'] == 'Placed' else 'None',
    axis=1
)

# Save the dataset
data.to_csv('/workspaces/pajaya/very_new_predict/Ultra_Predictive_Dataset.csv', index=False)

print("=== Ultra-Predictive Dataset Created ===")
print(f"Dataset shape: {data.shape}")
print("\nTarget distribution:")
print(data['Placement_Status'].value_counts(normalize=True))

# Calculate and display correlations with target
print("\nFeature correlations with target:")
target_numeric = pd.get_dummies(data['Placement_Status'], drop_first=True)['Placed']
correlations = []
for col in ['CGPA', 'Soft_Skills_Score', 'Technical_Skills', 'Leadership_Score', 
           'Experience_Years', 'No_of_Internships', 'Projects', 'Live_Backlogs',
           'No_of_Certifications']:
    corr = data[col].corr(target_numeric)
    correlations.append((col, corr))

# Sort and print correlations
for col, corr in sorted(correlations, key=lambda x: abs(x[1]), reverse=True):
    print(f"{col}: {corr:.3f}")

print("\nDataset saved as 'Ultra_Predictive_Dataset.csv'")
print("This dataset has EXTREMELY STRONG signal that should yield 95%+ accuracy.")
print("\nTo train models on this dataset, run:")
print("python /workspaces/pajaya/New_Predict/train_original.py")
print("(Make sure to update the path to use the new dataset in train_original.py)")
