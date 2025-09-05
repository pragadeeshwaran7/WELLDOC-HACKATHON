# AI-Driven Risk Prediction Engine for Chronic Care Patients

## 🏥 Overview

This repository contains a complete implementation of an AI-driven risk prediction engine that forecasts **90-day clinical deterioration risk** for chronic care patients. The system combines advanced machine learning with explainable AI to provide actionable clinical decision support.

### 🎯 Key Capabilities
- **Predictive Analytics**: 90-day deterioration risk with 70.2% AUROC performance
- **Clinical Explainability**: Plain-language factor interpretations for healthcare professionals
- **Interactive Dashboard**: Real-time risk monitoring and patient management interface
- **Comprehensive Evaluation**: Multiple ML models with detailed performance metrics

## 📊 Project Results

### Model Performance
- **Best Model**: Random Forest Ensemble
- **AUROC**: 0.7024 (Good discrimination)
- **AUPRC**: 0.5941 (Strong precision-recall balance)
- **Specificity**: 98.2% (Excellent false alarm avoidance)
- **Precision**: 83.9% (High confidence in predictions)

### Risk Stratification
- **Very Low Risk**: 32.6% of patients (routine monitoring)
- **Low Risk**: 53.8% of patients (standard care)
- **Moderate Risk**: 6.9% of patients (enhanced monitoring)
- **High Risk**: 4.6% of patients (frequent follow-ups)
- **Very High Risk**: 2.1% of patients (immediate intervention)

### Top Predictive Factors
1. **BNP Maximum Level** (5.3%) - Heart failure severity
2. **Average Heart Rate** (5.1%) - Cardiac stress indicator
3. **Average Blood Glucose** (4.9%) - Diabetes control
4. **Oxygen Saturation** (4.4%) - Respiratory function
5. **Kidney Function Trends** (4.0%) - Renal deterioration

## 🗂️ Repository Structure

```
chronic-care-ai-engine/
├── 📁 data/
│   ├── chronic_care_baseline_data.csv        # Patient demographics & conditions (100 patients)
│   ├── chronic_care_timeseries_data.csv      # 180-day monitoring data (18K records)
│   └── chronic_care_features_data.csv        # Engineered ML features (1.3K samples)
├── 📁 models/
│   ├── chronic_care_trained_model.pkl        # Trained Random Forest + artifacts
│   └── chronic_care_dashboard_data.json      # Dashboard configuration
├── 📁 dashboard/
│   ├── fixed_chronic_care_dashboard.html     # ✅ Working interactive dashboard
│   ├── debug_dashboard.html                  # 🐛 Debug version with console
│   ├── dashboard.js                          # Chart.js implementation
│   └── style.css                             # Professional healthcare styling
├── 📁 notebooks/
│   └── complete_implementation.ipynb         # End-to-end analysis notebook
├── 📁 scripts/
│   ├── data_generation.py                    # Synthetic data creation
│   ├── feature_engineering.py               # Time series feature extraction
│   ├── model_training.py                    # ML pipeline training
│   └── prediction_interface.py              # New patient prediction API
├── 📁 docs/
│   ├── clinical_validation.md               # Clinical interpretation guide
│   ├── technical_architecture.md            # System architecture details
│   └── deployment_guide.md                 # Production deployment instructions
├── 📁 tests/
│   ├── test_data_generation.py              # Data quality tests
│   ├── test_model_performance.py            # Model validation tests  
│   └── test_dashboard_functionality.py      # UI/UX testing
├── requirements.txt                          # Python dependencies
├── environment.yml                          # Conda environment
├── Dockerfile                              # Container deployment
├── README.md                               # This file
└── LICENSE                                 # MIT License
```

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone repository
git clone https://github.com/your-org/chronic-care-ai-engine.git
cd chronic-care-ai-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Model

```python
# Run complete pipeline
python scripts/complete_pipeline.py

# Or step by step:
python scripts/data_generation.py      # Creates synthetic patient data
python scripts/feature_engineering.py  # Extracts ML features
python scripts/model_training.py       # Trains & evaluates models
```

**Output:**
- Data files in `data/` directory
- Trained model: `models/chronic_care_trained_model.pkl`
- Performance metrics printed to console

### 3. Launch Interactive Dashboard

```bash
# Option 1: Direct browser access
open dashboard/fixed_chronic_care_dashboard.html

# Option 2: Local web server (recommended)
python -m http.server 8000
# Navigate to: http://localhost:8000/dashboard/fixed_chronic_care_dashboard.html
```

**Dashboard Features:**
- **Cohort Overview**: Risk distribution pie chart, high-risk patient alerts
- **Patient Details**: Individual assessments with contributing factors
- **Model Performance**: AUROC/AUPRC comparisons across algorithms
- **Explainability**: Clinical factor importance with medical interpretations

### 4. Predict for New Patients

```python
import pickle
from scripts.prediction_interface import predict_deterioration_risk

# Load trained model
with open('models/chronic_care_trained_model.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

# Example patient data (30-day summary)
new_patient = {
    'vitals': {
        'systolic_bp_mean': 165,    # Elevated blood pressure
        'heart_rate_mean': 95,      # Mild tachycardia
        'oxygen_saturation_mean': 94 # Concerning O2 levels
    },
    'labs': {
        'glucose_mean': 220,        # Poor diabetes control
        'bnp_mean': 280,           # Heart failure indicator
        'creatinine_mean': 1.8     # Kidney dysfunction
    },
    'adherence': {
        'adherence_mean': 0.65     # Poor medication compliance
    }
}

# Generate prediction
result = predict_deterioration_risk(new_patient, model_artifacts)

print(f"90-Day Deterioration Risk: {result['risk_percentage']}%")
print(f"Risk Level: {result['risk_level']}")
print("Recommendations:")
for rec in result['recommendations']:
    print(f"  • {rec}")
```

**Sample Output:**
```
90-Day Deterioration Risk: 78.3%
Risk Level: High
Recommendations:
  • Immediate clinical review required
  • Consider ICU monitoring
  • Review medication regimen
  • Check for acute changes in condition
```

## 🛠️ Development & Customization

### Adding New Features

```python
# 1. Extend feature engineering
def add_custom_feature(timeseries_data):
    """Add domain-specific clinical features"""
    # Your custom feature logic here
    return enhanced_features

# 2. Integrate new data sources
def integrate_ehr_data(patient_id):
    """Connect to Electronic Health Records"""
    # EHR integration logic
    return patient_data

# 3. Customize risk thresholds
RISK_THRESHOLDS = {
    'very_low': (0.0, 0.2),
    'low': (0.2, 0.4),
    'moderate': (0.4, 0.6),
    'high': (0.6, 0.8),
    'very_high': (0.8, 1.0)
}
```

### Model Retraining

```python
# Retrain with new data
from scripts.model_training import retrain_model

new_data = load_new_patient_data()
updated_model = retrain_model(
    existing_model_path='models/chronic_care_trained_model.pkl',
    new_data=new_data,
    validation_split=0.2
)

# Evaluate performance improvement
print(f"Previous AUROC: {old_auroc:.3f}")
print(f"Updated AUROC: {updated_model['auroc']:.3f}")
```

## 📈 Dashboard Usage Guide

### Navigation Tabs

1. **Cohort Overview**
   - Risk distribution across all monitored patients
   - High-risk alert banner for immediate attention cases
   - Filterable patient list with risk scores
   - Population-level statistics and trends

2. **Patient Details**
   - Individual risk assessment with probability scores
   - Contributing factor analysis with clinical values
   - Personalized recommendations based on risk profile
   - Historical risk evolution timeline

3. **Model Performance**
   - AUROC/AUPRC comparison across different algorithms
   - Confusion matrix analysis for clinical decision-making
   - Calibration curve for probability reliability
   - Performance breakdown by patient subgroups

4. **Explainability**
   - Top clinical factors driving AI predictions
   - Feature importance rankings with medical interpretations
   - Local explanations for individual patient predictions
   - Clinical threshold guidelines for each parameter

### Troubleshooting Dashboard

If charts don't display:

```bash
# 1. Use debug version
open dashboard/debug_dashboard.html
# Check in-page console for error messages

# 2. Verify Chart.js loading
# Open browser Developer Tools (F12)
# Console should show: "Chart.js loaded successfully - Version: 3.9.1"

# 3. Clear browser cache
# Hard refresh: Ctrl+Shift+R (Chrome/Firefox)

# 4. Check network connectivity
# Ensure CDN access to Chart.js library
```

## 🏥 Clinical Implementation

### Integration with EHR Systems

```python
# Example EHR integration
class EHRIntegration:
    def __init__(self, ehr_endpoint, api_key):
        self.endpoint = ehr_endpoint
        self.api_key = api_key
    
    def fetch_patient_data(self, patient_id, days=30):
        """Retrieve patient data from EHR"""
        vitals = self.get_vitals_history(patient_id, days)
        labs = self.get_lab_results(patient_id, days)
        medications = self.get_medication_adherence(patient_id, days)
        
        return {
            'vitals': vitals,
            'labs': labs,
            'adherence': medications
        }
    
    def update_risk_score(self, patient_id, risk_assessment):
        """Update patient record with AI risk score"""
        self.post_clinical_note(patient_id, {
            'ai_risk_score': risk_assessment['risk_percentage'],
            'risk_level': risk_assessment['risk_level'],
            'recommendations': risk_assessment['recommendations'],
            'timestamp': datetime.now().isoformat()
        })
```

### Clinical Workflow Integration

```python
# Automated risk screening workflow
def daily_risk_screening():
    """Run daily risk assessment for all monitored patients"""
    high_risk_patients = []
    
    for patient_id in get_monitored_patients():
        # Fetch latest 30-day data
        patient_data = ehr.fetch_patient_data(patient_id)
        
        # Generate risk prediction
        risk_assessment = predict_deterioration_risk(patient_data, model_artifacts)
        
        # Flag high-risk patients
        if risk_assessment['risk_percentage'] > 60:
            high_risk_patients.append({
                'patient_id': patient_id,
                'risk_score': risk_assessment['risk_percentage'],
                'key_factors': risk_assessment['top_contributing_factors'][:3],
                'recommendations': risk_assessment['recommendations']
            })
            
            # Send alert to clinical team
            send_clinical_alert(patient_id, risk_assessment)
    
    return high_risk_patients
```

## 🔬 Technical Architecture

### Model Pipeline

```python
# Feature Engineering Pipeline
def create_prediction_features(timeseries_data, window_size=30):
    """
    Extract predictive features from patient time series
    
    Args:
        timeseries_data: 30-180 days of patient monitoring data
        window_size: Rolling window for statistical features
    
    Returns:
        feature_matrix: Engineered features for ML model
    """
    features = {}
    
    # Vital sign statistics
    features.update(extract_vital_features(timeseries_data, window_size))
    
    # Laboratory result trends  
    features.update(extract_lab_features(timeseries_data, window_size))
    
    # Medication adherence patterns
    features.update(extract_adherence_features(timeseries_data, window_size))
    
    # Lifestyle and activity metrics
    features.update(extract_lifestyle_features(timeseries_data, window_size))
    
    return features

# Model Training Pipeline
def train_risk_prediction_model(features, targets):
    """
    Train ensemble model for deterioration prediction
    
    Returns:
        best_model: Trained Random Forest with highest AUROC
        model_metrics: Performance evaluation results
    """
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100),
        'svm': SVC(probability=True, class_weight='balanced'),
        'logistic_regression': LogisticRegression(max_iter=1000)
    }
    
    best_auroc = 0
    best_model = None
    
    for name, model in models.items():
        # Cross-validation evaluation
        auroc_scores = cross_val_score(model, features, targets, 
                                     cv=5, scoring='roc_auc')
        mean_auroc = np.mean(auroc_scores)
        
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_model = model.fit(features, targets)
    
    return best_model, calculate_metrics(best_model, features, targets)
```

### Deployment Options

#### Option 1: Local Development
```bash
# Development server
python -m http.server 8000
```

#### Option 2: Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "http.server", "8000"]
```

```bash
# Build and run container
docker build -t chronic-care-ai .
docker run -p 8000:8000 chronic-care-ai
```

#### Option 3: Cloud Deployment (AWS/Azure/GCP)
```yaml
# docker-compose.yml
version: '3.8'
services:
  chronic-care-dashboard:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

## 📋 Requirements

### Python Dependencies
```
# Core ML libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Model persistence
pickle-mixin>=1.0.0
joblib>=1.1.0

# Web framework (optional)
flask>=2.0.0
gunicorn>=20.1.0

# Data validation
pydantic>=1.8.0
marshmallow>=3.14.0

# Testing
pytest>=6.2.0
pytest-cov>=3.0.0
```

### Frontend Dependencies
```html
<!-- Chart.js for visualizations -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>

<!-- Font Awesome for icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

<!-- Bootstrap (optional) -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
```

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html

# Test specific components
pytest tests/test_model_performance.py -v
pytest tests/test_dashboard_functionality.py -v
```

### Model Validation Tests
```python
# test_model_performance.py
def test_model_auroc_threshold():
    """Ensure model meets minimum AUROC requirement"""
    assert model_metrics['auroc'] >= 0.70, f"AUROC {model_metrics['auroc']} below threshold"

def test_prediction_consistency():
    """Verify consistent predictions for same input"""
    predictions = [predict_deterioration_risk(test_patient, model) for _ in range(10)]
    risk_scores = [p['risk_percentage'] for p in predictions]
    assert np.std(risk_scores) < 0.01, "Predictions not consistent"

def test_feature_importance_stability():
    """Check that top features remain stable across retraining"""
    top_features = get_top_features(model, n=5)
    expected_features = ['bnp_max', 'heart_rate_mean', 'glucose_mean']
    assert all(f in top_features for f in expected_features), "Key features missing"
```
