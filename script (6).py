# Fix JSON serialization and create dashboard data
import json

# Convert numpy types to Python types for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Create comprehensive dashboard data with proper type conversion
dashboard_data = {
    'model_performance': {
        'auroc': float(0.7024),
        'auprc': float(0.5941),
        'sensitivity': float(0.2342),
        'specificity': float(0.9821),
        'precision': float(0.8387),
        'accuracy': float(0.8154)
    },
    'patient_cohort': {
        'total_patients': int(len(features_df['patient_id'].unique())),
        'total_predictions': int(len(features_df)),
        'positive_cases': int(y_test.sum()),
        'negative_cases': int(len(y_test) - y_test.sum())
    },
    'risk_distribution': {
        'very_low': int((y_pred_proba_rf <= 0.2).sum()),
        'low': int(((y_pred_proba_rf > 0.2) & (y_pred_proba_rf <= 0.4)).sum()),
        'moderate': int(((y_pred_proba_rf > 0.4) & (y_pred_proba_rf <= 0.6)).sum()),
        'high': int(((y_pred_proba_rf > 0.6) & (y_pred_proba_rf <= 0.8)).sum()),
        'very_high': int((y_pred_proba_rf > 0.8).sum())
    },
    'feature_importance': [
        {
            'feature': str(row['feature']),
            'importance': float(row['importance']),
            'clinical_name': str(row['feature']).replace('_', ' ').title()
        } for _, row in feature_importance.head(15).iterrows()
    ],
    'calibration': {
        'mean_predicted_value': [float(x) for x in mean_predicted_value.tolist()],
        'fraction_of_positives': [float(x) for x in fraction_of_positives.tolist()]
    },
    'clinical_explanations': clinical_explanations,
    'sample_patients': []
}

# Add sample patient data for dashboard
for i in range(min(10, len(X_test))):
    patient_features = X_test.iloc[i]
    predicted_risk = float(y_pred_proba_rf[i])
    actual_outcome = bool(y_test.iloc[i])
    
    # Determine risk level
    if predicted_risk > 0.8:
        risk_level = "Very High"
        risk_color = "#dc3545"  # Red
    elif predicted_risk > 0.6:
        risk_level = "High"
        risk_color = "#fd7e14"  # Orange
    elif predicted_risk > 0.4:
        risk_level = "Moderate"
        risk_color = "#ffc107"  # Yellow
    elif predicted_risk > 0.2:
        risk_level = "Low"
        risk_color = "#28a745"  # Green
    else:
        risk_level = "Very Low"
        risk_color = "#6c757d"  # Gray
    
    # Get key risk factors
    key_factors = []
    for _, row in feature_importance.head(5).iterrows():
        feature = row['feature']
        value = float(patient_features[feature])
        key_factors.append({
            'name': str(feature).replace('_', ' ').title(),
            'value': round(value, 2),
            'feature': feature
        })
    
    patient_data = {
        'patient_id': f"Patient_{i+1:03d}",
        'predicted_risk': round(predicted_risk, 3),
        'risk_percentage': round(predicted_risk * 100, 1),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'actual_outcome': actual_outcome,
        'key_factors': key_factors
    }
    
    dashboard_data['sample_patients'].append(patient_data)

# Apply numpy type conversion
dashboard_data = convert_numpy_types(dashboard_data)

# Save dashboard data as JSON
with open('dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print("Dashboard data saved successfully!")

# Display dashboard summary
print("\n" + "="*60)
print("DASHBOARD DATA SUMMARY")
print("="*60)

print(f"Model Performance:")
for key, value in dashboard_data['model_performance'].items():
    print(f"  {key.upper()}: {value:.4f}")

print(f"\nPatient Cohort:")
for key, value in dashboard_data['patient_cohort'].items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

print(f"\nRisk Distribution:")
total_patients = sum(dashboard_data['risk_distribution'].values())
for level, count in dashboard_data['risk_distribution'].items():
    percentage = (count / total_patients) * 100 if total_patients > 0 else 0
    print(f"  {level.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

print(f"\nTop 5 Most Important Clinical Factors:")
for factor in dashboard_data['feature_importance'][:5]:
    print(f"  {factor['clinical_name']}: {factor['importance']:.4f}")

print(f"\nSample High-Risk Patients:")
high_risk_patients = [p for p in dashboard_data['sample_patients'] if p['predicted_risk'] > 0.6]
for patient in high_risk_patients[:3]:
    print(f"  {patient['patient_id']}: {patient['risk_percentage']}% risk ({patient['risk_level']})")

print(f"\nFiles created for dashboard:")
print(f"  - dashboard_data.json: Complete dashboard configuration")
print(f"  - chronic_care_model.pkl: Trained ML model")
print(f"  - Patient data CSVs: Baseline, timeseries, and features")

# Create model architecture summary for documentation
model_architecture = {
    "prediction_approach": "Ensemble Learning with Random Forest as Best Performer",
    "input_features": {
        "vital_signs": ["systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature", "oxygen_saturation"],
        "lab_results": ["glucose", "hba1c", "creatinine", "bnp"],
        "medication_adherence": ["medication_adherence_mean", "medication_adherence_trend"],
        "lifestyle_factors": ["steps_per_day", "sleep_hours"]
    },
    "feature_engineering": {
        "temporal_window": "30 days of historical data",
        "statistical_features": ["mean", "std", "trend", "min", "max"],
        "prediction_horizon": "90 days ahead"
    },
    "model_types_evaluated": ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "Ensemble"],
    "best_model": "Random Forest",
    "evaluation_metrics": ["AUROC", "AUPRC", "Sensitivity", "Specificity", "Precision", "Calibration"],
    "explainability_methods": ["Feature Importance", "Clinical Rule-Based Explanations"]
}

with open('model_architecture.json', 'w') as f:
    json.dump(model_architecture, f, indent=2)

print(f"  - model_architecture.json: Technical documentation")
print(f"\nAll files ready for dashboard development!")