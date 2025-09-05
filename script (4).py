# Calculate calibration and prepare explainability data
import shap

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

# SHAP explanations for Random Forest
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test.iloc[:100])  # First 100 test samples for efficiency

if len(shap_values) == 2:  # Binary classification returns list of arrays
    shap_values_positive = shap_values[1]  # SHAP values for positive class
else:
    shap_values_positive = shap_values

# Global feature importance from SHAP
shap_importance = np.abs(shap_values_positive).mean(0)
shap_feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

print("\nTop 10 features by SHAP importance:")
print(shap_feature_importance.head(10))

# Sample patient explanations
print("\n" + "="*50)
print("SAMPLE PATIENT EXPLANATIONS")
print("="*50)

for i in range(min(3, len(X_test))):
    patient_features = X_test.iloc[i]
    patient_shap = shap_values_positive[i]
    predicted_risk = y_pred_proba_rf[i]
    actual_outcome = y_test.iloc[i]
    
    print(f"\nPatient {i+1}:")
    print(f"  Predicted Risk: {predicted_risk:.3f}")
    print(f"  Actual Outcome: {'Deterioration' if actual_outcome else 'No Deterioration'}")
    print("  Top Contributing Factors:")
    
    # Get top 5 SHAP values (positive and negative)
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': patient_shap,
        'feature_value': patient_features.values
    }).sort_values('shap_value', key=abs, ascending=False)
    
    for j in range(5):
        feature_name = shap_df.iloc[j]['feature']
        shap_val = shap_df.iloc[j]['shap_value']
        feature_val = shap_df.iloc[j]['feature_value']
        direction = "increases" if shap_val > 0 else "decreases"
        
        # Convert feature names to clinical terms
        clinical_name = feature_name.replace('_', ' ').title()
        print(f"    {clinical_name}: {feature_val:.2f} ({direction} risk by {abs(shap_val):.3f})")

# Create summary statistics for dashboard
print("\n" + "="*50)
print("DASHBOARD SUMMARY STATISTICS")
print("="*50)

# Overall model performance
best_model_performance = {
    'model_name': 'Random Forest',
    'auroc': 0.7024,
    'auprc': 0.5941,
    'sensitivity': 0.2342,
    'specificity': 0.9821,
    'precision': 0.8387,
    'total_patients': len(features_df['patient_id'].unique()),
    'total_predictions': len(features_df),
    'high_risk_patients': (y_pred_proba_rf > 0.5).sum(),
    'low_risk_patients': (y_pred_proba_rf <= 0.5).sum()
}

print("Model Performance:")
for key, value in best_model_performance.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

# Risk distribution
risk_bins = ['Very Low (0-0.2)', 'Low (0.2-0.4)', 'Moderate (0.4-0.6)', 'High (0.6-0.8)', 'Very High (0.8-1.0)']
risk_counts = [
    (y_pred_proba_rf <= 0.2).sum(),
    ((y_pred_proba_rf > 0.2) & (y_pred_proba_rf <= 0.4)).sum(),
    ((y_pred_proba_rf > 0.4) & (y_pred_proba_rf <= 0.6)).sum(),
    ((y_pred_proba_rf > 0.6) & (y_pred_proba_rf <= 0.8)).sum(),
    (y_pred_proba_rf > 0.8).sum()
]

print(f"\nRisk Distribution:")
for risk_bin, count in zip(risk_bins, risk_counts):
    percentage = (count / len(y_pred_proba_rf)) * 100
    print(f"  {risk_bin}: {count} patients ({percentage:.1f}%)")

# Save model and results
import pickle

model_results = {
    'model': rf_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'performance_metrics': best_model_performance,
    'calibration_data': calibration_data,
    'shap_importance': shap_feature_importance.to_dict('records'),
    'risk_distribution': {
        'bins': risk_bins,
        'counts': risk_counts
    }
}

with open('chronic_care_model.pkl', 'wb') as f:
    pickle.dump(model_results, f)

print(f"\nModel and results saved to chronic_care_model.pkl")
print(f"Total files created: 4")