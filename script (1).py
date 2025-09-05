# Generate time series data for vitals, labs, and medication adherence
import random
from datetime import datetime, timedelta

# Create time series data for each patient over 180 days
start_date = datetime.now() - timedelta(days=180)
dates = [start_date + timedelta(days=x) for x in range(180)]

patient_timeseries = []

for _, patient in baseline_df.iterrows():
    patient_id = patient['patient_id']
    age = patient['age']
    conditions = patient['chronic_conditions'].split(';')
    baseline_risk = patient['baseline_risk_score']
    
    # Determine if patient will deteriorate (outcome)
    deterioration_risk = baseline_risk + 0.1 * (age - 50) / 35
    if 'Heart_Failure' in conditions: deterioration_risk += 0.2
    if 'Diabetes' in conditions: deterioration_risk += 0.15
    if 'COPD' in conditions: deterioration_risk += 0.18
    
    will_deteriorate = np.random.random() < min(deterioration_risk, 0.8)
    deterioration_day = np.random.randint(120, 180) if will_deteriorate else None
    
    for day_idx, date in enumerate(dates):
        # Progressive deterioration if patient will deteriorate
        if will_deteriorate and deterioration_day and day_idx > deterioration_day - 30:
            trend_factor = (day_idx - (deterioration_day - 30)) / 30
        else:
            trend_factor = 0
        
        # Vital signs with deterioration trends
        base_systolic = 130 if 'Hypertension' in conditions else 120
        systolic_bp = base_systolic + np.random.normal(0, 15) + trend_factor * 25
        
        base_diastolic = 85 if 'Hypertension' in conditions else 80
        diastolic_bp = base_diastolic + np.random.normal(0, 10) + trend_factor * 15
        
        heart_rate = 75 + np.random.normal(0, 12) + trend_factor * 20
        
        respiratory_rate = 18 + np.random.normal(0, 3) + trend_factor * 8
        
        temperature = 98.6 + np.random.normal(0, 1) + trend_factor * 2
        
        oxygen_saturation = 98 - np.random.normal(0, 2) - trend_factor * 5
        
        # Lab values with condition-specific patterns
        if 'Diabetes' in conditions:
            base_glucose = 140
        else:
            base_glucose = 100
        glucose = base_glucose + np.random.normal(0, 30) + trend_factor * 60
        
        hba1c = 7.2 + np.random.normal(0, 0.8) + trend_factor * 1.5 if 'Diabetes' in conditions else 5.5 + np.random.normal(0, 0.3)
        
        creatinine = 1.1 + np.random.normal(0, 0.3) + trend_factor * 0.8
        
        if 'Heart_Failure' in conditions:
            bnp = 250 + np.random.normal(0, 100) + trend_factor * 400
        else:
            bnp = 50 + np.random.normal(0, 30)
            
        # Medication adherence (decreases before deterioration)
        base_adherence = 0.85
        if will_deteriorate and deterioration_day and day_idx > deterioration_day - 20:
            adherence_factor = -0.3 * ((day_idx - (deterioration_day - 20)) / 20)
        else:
            adherence_factor = 0
        
        medication_adherence = max(0, min(1, base_adherence + np.random.normal(0, 0.1) + adherence_factor))
        
        # Activity and lifestyle data
        steps_per_day = max(0, 4000 + np.random.normal(0, 1500) - trend_factor * 2000)
        sleep_hours = max(0, 7.5 + np.random.normal(0, 1.2) - trend_factor * 2)
        
        patient_timeseries.append({
            'patient_id': patient_id,
            'date': date.strftime('%Y-%m-%d'),
            'day_from_start': day_idx,
            'systolic_bp': max(80, systolic_bp),
            'diastolic_bp': max(50, diastolic_bp),
            'heart_rate': max(40, heart_rate),
            'respiratory_rate': max(8, respiratory_rate),
            'temperature': temperature,
            'oxygen_saturation': max(80, min(100, oxygen_saturation)),
            'glucose': max(70, glucose),
            'hba1c': max(4, hba1c),
            'creatinine': max(0.5, creatinine),
            'bnp': max(0, bnp),
            'medication_adherence': medication_adherence,
            'steps_per_day': steps_per_day,
            'sleep_hours': sleep_hours,
            'will_deteriorate': will_deteriorate,
            'deterioration_day': deterioration_day if deterioration_day else -1
        })

timeseries_df = pd.DataFrame(patient_timeseries)
print("Time Series Data Sample:")
print(timeseries_df.head(10))
print(f"Total records: {len(timeseries_df)}")
print(f"Patients who will deteriorate: {len(timeseries_df[timeseries_df['will_deteriorate'] == True]['patient_id'].unique())}")