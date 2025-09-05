# Calculate calibration and create explainability without SHAP
from sklearn.calibration import calibration_curve
import pickle

# Calculate calibration for best model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba_rf, n_bins=10)

calibration_data = {
    'mean_predicted_value': mean_predicted_value.tolist(),
    'fraction_of_positives': fraction_of_positives.tolist()
}

print("Calibration curve data:")
for i in range(len(mean_predicted_value)):
    print(f"  Predicted: {mean_predicted_value[i]:.3f}, Actual: {fraction_of_positives[i]:.3f}")

# Alternative explainability using feature importance and correlation analysis
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 features by importance:")
print(feature_importance.head(15))

# Create clinical explanations for top features
clinical_explanations = {
    'bnp_max': 'Maximum B-type Natriuretic Peptide - indicates heart failure severity',
    'heart_rate_mean': 'Average heart rate - elevated rates may indicate cardiac stress',
    'glucose_mean': 'Average blood glucose - poor diabetes control increases risk',
    'oxygen_saturation_mean': 'Average oxygen saturation - declining levels indicate respiratory issues',
    'bnp_mean': 'Average B-type Natriuretic Peptide - consistently elevated indicates heart failure',
    'creatinine_trend': 'Kidney function trend - worsening indicates renal deterioration',
    'creatinine_mean': 'Average kidney function - elevated indicates kidney disease progression',
    'hba1c_latest': 'Latest diabetes control marker - higher values indicate poor control',
    'respiratory_rate_std': 'Respiratory rate variability - high variability indicates instability',
    'oxygen_saturation_min': 'Minimum oxygen saturation - low values indicate severe respiratory issues',
    'medication_adherence_mean': 'Average medication compliance - low adherence increases risk',
    'medication_adherence_trend': 'Medication compliance trend - declining adherence is concerning',
    'systolic_bp_trend': 'Blood pressure trend - rapidly changing BP indicates instability',
    'glucose_trend': 'Blood glucose trend - rapidly changing glucose indicates poor control',
    'temperature_mean': 'Average body temperature - elevated may indicate infection'
}

print(f"\nClinical Interpretations:")
for _, row in feature_importance.head(10).iterrows():
    feature = row['feature']
    importance = row['importance']
    explanation = clinical_explanations.get(feature, 'Clinical parameter affecting patient risk')
    print(f"  {feature}: {explanation} (Importance: {importance:.3f})")

# Sample patient risk assessments with explanations
print("\n" + "="*50)
print("SAMPLE PATIENT RISK ASSESSMENTS")
print("="*50)

for i in range(min(5, len(X_test))):
    patient_features = X_test.iloc[i]
    predicted_risk = y_pred_proba_rf[i]
    actual_outcome = y_test.iloc[i]
    
    print(f"\nPatient {i+1}:")
    print(f"  Predicted 90-day Deterioration Risk: {predicted_risk:.1%}")
    print(f"  Actual Outcome: {'Deterioration' if actual_outcome else 'No Deterioration'}")
    
    # Get top contributing factors based on feature importance and values
    risk_factors = []
    for _, row in feature_importance.head(10).iterrows():
        feature = row['feature']
        importance = row['importance']
        value = patient_features[feature]
        
        # Determine if value is concerning based on feature type
        is_concerning = False
        if 'bnp' in feature and value > 150:
            is_concerning = True
        elif 'glucose' in feature and value > 180:
            is_concerning = True
        elif 'creatinine' in feature and value > 1.5:
            is_concerning = True
        elif 'heart_rate' in feature and value > 100:
            is_concerning = True
        elif 'respiratory_rate' in feature and value > 24:
            is_concerning = True
        elif 'oxygen_saturation' in feature and value < 95:
            is_concerning = True
        elif 'medication_adherence' in feature and value < 0.8:
            is_concerning = True
        elif 'systolic_bp' in feature and value > 160:
            is_concerning = True
        
        if is_concerning:
            risk_factors.append({
                'feature': feature,
                'value': value,
                'importance': importance,
                'explanation': clinical_explanations.get(feature, 'Clinical parameter')
            })
    
    risk_factors.sort(key=lambda x: x['importance'], reverse=True)
    
    print("  Key Risk Factors:")
    if risk_factors:
        for factor in risk_factors[:3]:
            print(f"    - {factor['feature']}: {factor['value']:.2f}")
            print(f"      {factor['explanation']}")
    else:
        print("    - No major risk factors identified")
    
    # Recommendations
    print("  Recommended Actions:")
    if predicted_risk > 0.7:
        print("    - Immediate clinical review required")
        print("    - Consider ICU monitoring")
        print("    - Review medication regimen")
    elif predicted_risk > 0.4:
        print("    - Increased monitoring frequency")
        print("    - Schedule follow-up within 48-72 hours")
        print("    - Review medication adherence")
    else:
        print("    - Continue routine monitoring")
        print("    - Standard care protocols")

# Create comprehensive dashboard data
dashboard_data = {
    'model_performance': {
        'auroc': 0.7024,
        'auprc': 0.5941,
        'sensitivity': 0.2342,
        'specificity': 0.9821,
        'precision': 0.8387
    },
    'patient_cohort': {
        'total_patients': len(features_df['patient_id'].unique()),
        'total_predictions': len(features_df),
        'positive_cases': y_test.sum(),
        'negative_cases': len(y_test) - y_test.sum()
    },
    'risk_distribution': {
        'very_low': int((y_pred_proba_rf <= 0.2).sum()),
        'low': int(((y_pred_proba_rf > 0.2) & (y_pred_proba_rf <= 0.4)).sum()),
        'moderate': int(((y_pred_proba_rf > 0.4) & (y_pred_proba_rf <= 0.6)).sum()),
        'high': int(((y_pred_proba_rf > 0.6) & (y_pred_proba_rf <= 0.8)).sum()),
        'very_high': int((y_pred_proba_rf > 0.8).sum())
    },
    'feature_importance': feature_importance.head(15).to_dict('records'),
    'calibration': calibration_data,
    'clinical_explanations': clinical_explanations
}

# Save model and dashboard data
model_results = {
    'model': rf_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'dashboard_data': dashboard_data
}

with open('chronic_care_model.pkl', 'wb') as f:
    pickle.dump(model_results, f)

# Save dashboard data as JSON for web app
import json
with open('dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print(f"\nModel and dashboard data saved!")
print(f"Files created:")
print(f"  - chronic_care_model.pkl (trained model)")
print(f"  - dashboard_data.json (dashboard data)")
print(f"  - chronic_care_baseline.csv (patient baseline data)")
print(f"  - chronic_care_timeseries.csv (time series data)")
print(f"  - chronic_care_features.csv (engineered features)")

print(f"\nDashboard Summary:")
print(f"  Total patients monitored: {dashboard_data['patient_cohort']['total_patients']}")
print(f"  High-risk patients: {dashboard_data['risk_distribution']['high'] + dashboard_data['risk_distribution']['very_high']}")
print(f"  Model accuracy (AUROC): {dashboard_data['model_performance']['auroc']:.3f}")
print(f"  Model precision: {dashboard_data['model_performance']['precision']:.3f}")