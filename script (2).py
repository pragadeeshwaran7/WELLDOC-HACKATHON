# Save the datasets and create evaluation metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Save datasets
baseline_df.to_csv('chronic_care_baseline.csv', index=False)
timeseries_df.to_csv('chronic_care_timeseries.csv', index=False)

print("Datasets saved!")
print(f"Baseline data: {baseline_df.shape}")
print(f"Time series data: {timeseries_df.shape}")

# Prepare feature engineering for prediction model
def create_features(df, days_window=30):
    """Create features from time series data for prediction"""
    features = []
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].sort_values('day_from_start')
        
        # For each prediction point (day 30-150), create features from previous 30 days
        for pred_day in range(30, 151, 10):  # Every 10 days for efficiency
            if pred_day >= len(patient_data):
                continue
                
            # Get window of data (30 days before prediction point)
            window_data = patient_data[patient_data['day_from_start'] < pred_day].tail(days_window)
            
            if len(window_data) < days_window:
                continue
            
            # Extract features
            feature_dict = {
                'patient_id': patient_id,
                'prediction_day': pred_day,
                
                # Vital signs - mean, std, trend
                'systolic_bp_mean': window_data['systolic_bp'].mean(),
                'systolic_bp_std': window_data['systolic_bp'].std(),
                'systolic_bp_trend': np.polyfit(range(len(window_data)), window_data['systolic_bp'], 1)[0],
                
                'diastolic_bp_mean': window_data['diastolic_bp'].mean(),
                'diastolic_bp_std': window_data['diastolic_bp'].std(),
                'diastolic_bp_trend': np.polyfit(range(len(window_data)), window_data['diastolic_bp'], 1)[0],
                
                'heart_rate_mean': window_data['heart_rate'].mean(),
                'heart_rate_std': window_data['heart_rate'].std(),
                'heart_rate_trend': np.polyfit(range(len(window_data)), window_data['heart_rate'], 1)[0],
                
                'respiratory_rate_mean': window_data['respiratory_rate'].mean(),
                'respiratory_rate_std': window_data['respiratory_rate'].std(),
                'respiratory_rate_trend': np.polyfit(range(len(window_data)), window_data['respiratory_rate'], 1)[0],
                
                'temperature_mean': window_data['temperature'].mean(),
                'temperature_std': window_data['temperature'].std(),
                
                'oxygen_saturation_mean': window_data['oxygen_saturation'].mean(),
                'oxygen_saturation_min': window_data['oxygen_saturation'].min(),
                
                # Lab values
                'glucose_mean': window_data['glucose'].mean(),
                'glucose_std': window_data['glucose'].std(),
                'glucose_trend': np.polyfit(range(len(window_data)), window_data['glucose'], 1)[0],
                
                'hba1c_latest': window_data['hba1c'].iloc[-1],
                'creatinine_mean': window_data['creatinine'].mean(),
                'creatinine_trend': np.polyfit(range(len(window_data)), window_data['creatinine'], 1)[0],
                
                'bnp_mean': window_data['bnp'].mean(),
                'bnp_max': window_data['bnp'].max(),
                
                # Medication adherence
                'medication_adherence_mean': window_data['medication_adherence'].mean(),
                'medication_adherence_trend': np.polyfit(range(len(window_data)), window_data['medication_adherence'], 1)[0],
                
                # Lifestyle
                'steps_mean': window_data['steps_per_day'].mean(),
                'steps_trend': np.polyfit(range(len(window_data)), window_data['steps_per_day'], 1)[0],
                'sleep_mean': window_data['sleep_hours'].mean(),
            }
            
            # Target: will deteriorate within next 90 days
            future_data = patient_data[patient_data['day_from_start'] >= pred_day]
            target_window = future_data.head(90)
            
            will_deteriorate_90d = False
            if len(target_window) > 0:
                deterioration_day = patient_data['deterioration_day'].iloc[0]
                if deterioration_day > 0 and deterioration_day >= pred_day and deterioration_day < pred_day + 90:
                    will_deteriorate_90d = True
            
            feature_dict['target'] = will_deteriorate_90d
            features.append(feature_dict)
    
    return pd.DataFrame(features)

print("Creating features...")
features_df = create_features(timeseries_df)
print(f"Features created: {features_df.shape}")
print(f"Positive cases (deterioration): {features_df['target'].sum()}")
print(f"Negative cases: {(~features_df['target']).sum()}")

# Save features
features_df.to_csv('chronic_care_features.csv', index=False)
print("Features saved!")