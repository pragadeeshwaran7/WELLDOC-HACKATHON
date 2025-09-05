import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Create synthetic chronic care patient data for demonstration
np.random.seed(42)

# Patient demographics and baseline characteristics
n_patients = 100
patient_ids = [f"PAT_{i:03d}" for i in range(1, n_patients + 1)]

# Generate patient baseline data
baseline_data = []
for patient_id in patient_ids:
    age = np.random.randint(45, 85)
    gender = np.random.choice(['M', 'F'])
    
    # Chronic conditions (can have multiple)
    conditions = []
    if np.random.random() < 0.4: conditions.append('Diabetes')
    if np.random.random() < 0.3: conditions.append('Heart_Failure')
    if np.random.random() < 0.35: conditions.append('Hypertension')
    if np.random.random() < 0.2: conditions.append('COPD')
    if len(conditions) == 0: conditions.append('Diabetes')  # Ensure at least one condition
    
    baseline_data.append({
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'chronic_conditions': ';'.join(conditions),
        'bmi': np.random.normal(28, 5),  # Higher BMI for chronic patients
        'baseline_risk_score': np.random.random() * 0.3  # Base risk
    })

baseline_df = pd.DataFrame(baseline_data)
print("Baseline Patient Data:")
print(baseline_df.head())
print(f"Total patients: {len(baseline_df)}")