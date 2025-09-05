# Train prediction models and calculate evaluation metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Prepare data for modeling
feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'prediction_day', 'target']]
X = features_df[feature_cols].fillna(0)  # Handle any missing values
y = features_df['target'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training positives: {y_train.sum()}")
print(f"Test positives: {y_test.sum()}")

# Define base models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
}

# Train models and collect results
results = {}
model_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name in ['Logistic Regression', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Confusion Matrix': cm,
        'Classification Report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    model_predictions[name] = {
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }
    
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")

# Create ensemble model (Stacking)
print(f"\nTraining Ensemble Model (Voting)...")
voting_model = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='soft'
)

# Fit ensemble with both scaled and unscaled features (using preprocessing internally)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Create ensemble pipeline
ensemble = Pipeline([
    ('scaler', StandardScaler()),
    ('voting', VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ],
        voting='soft'
    ))
])

ensemble.fit(X_train, y_train)
y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
y_pred_ensemble = ensemble.predict(X_test)

auroc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
auprc_ensemble = average_precision_score(y_test, y_pred_proba_ensemble)
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)

results['Ensemble'] = {
    'AUROC': auroc_ensemble,
    'AUPRC': auprc_ensemble,
    'Confusion Matrix': cm_ensemble,
    'Classification Report': classification_report(y_test, y_pred_ensemble, output_dict=True)
}

print(f"  AUROC: {auroc_ensemble:.4f}")
print(f"  AUPRC: {auprc_ensemble:.4f}")

# Display results summary
print("\n" + "="*60)
print("CHRONIC CARE DETERIORATION PREDICTION RESULTS")
print("="*60)

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  AUROC: {metrics['AUROC']:.4f}")
    print(f"  AUPRC: {metrics['AUPRC']:.4f}")
    
    tn, fp, fn, tp = metrics['Confusion Matrix'].ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")

# Feature importance for Random Forest
print("\n" + "="*40)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*40)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))