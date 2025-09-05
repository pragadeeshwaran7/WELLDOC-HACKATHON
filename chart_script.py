import plotly.graph_objects as go
import plotly.io as pio

# Data
models = ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "Ensemble"]
auroc = [0.6132, 0.7024, 0.6685, 0.7066, 0.6928]
auprc = [0.4491, 0.5941, 0.5440, 0.5608, 0.5808]

# Abbreviate model names to fit within 15 character limit
model_abbrev = ["Log Regression", "Random Forest", "Grad Boosting", "SVM", "Ensemble"]

# Create figure
fig = go.Figure()

# Add AUROC bars
fig.add_trace(go.Bar(
    name='AUROC',
    x=model_abbrev,
    y=auroc,
    marker_color='#1FB8CD',
    text=[f'{val:.4f}' for val in auroc],
    textposition='outside'
))

# Add AUPRC bars
fig.add_trace(go.Bar(
    name='AUPRC',
    x=model_abbrev,
    y=auprc,
    marker_color='#DB4545',
    text=[f'{val:.4f}' for val in auprc],
    textposition='outside'
))

# Add horizontal line at 0.7 to show good performance threshold
fig.add_hline(y=0.7, line_dash="dash", line_color="gray")

# Update layout
fig.update_layout(
    title="AI Model Performance: Chronic Care",
    xaxis_title="ML Models",
    yaxis_title="Perf Score",
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    yaxis=dict(range=[0, 1])
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("model_performance_comparison.png")