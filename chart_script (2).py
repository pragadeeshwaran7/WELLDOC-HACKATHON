import plotly.graph_objects as go
import numpy as np

# Data from the provided JSON
clinical_factors = [
    "BNP Maximum Level",
    "Average Heart Rate", 
    "Average Blood Glucose",
    "Average Oxygen Saturation",
    "Average BNP Level",
    "Kidney Function Trend",
    "Average Kidney Function",
    "Latest Diabetes Control (HbA1c)",
    "Respiratory Rate Variability",
    "Minimum Oxygen Saturation"
]

importance_scores = [0.0530, 0.0507, 0.0486, 0.0440, 0.0416, 0.0401, 0.0399, 0.0372, 0.0370, 0.0367]

# Abbreviate clinical factor names to fit 15 character limit
abbreviated_factors = [
    "BNP Max Level",
    "Avg Heart Rate", 
    "Avg Blood Gluc",
    "Avg O2 Sat",
    "Avg BNP Level",
    "Kidney Trend",
    "Avg Kidney Func",
    "Latest HbA1c",
    "Resp Rate Var",
    "Min O2 Sat"
]

# Create horizontal bar chart
fig = go.Figure()

# Create gradient colors from light blue to dark blue (highest importance gets darkest blue)
# Since data is already sorted by importance (highest first), first bar should be darkest
gradient_colors = [
    '#0047AB',  # Dark blue for highest importance
    '#0066CC',
    '#0080FF', 
    '#1E90FF',
    '#4169E1',
    '#6495ED',
    '#87CEEB',
    '#ADD8E6',
    '#B0E0E6',
    '#E0F6FF'   # Light blue for lowest importance
]

fig.add_trace(go.Bar(
    x=importance_scores,
    y=abbreviated_factors,
    orientation='h',
    marker=dict(
        color=gradient_colors,
        line=dict(color='rgba(0,0,0,0)', width=0)
    ),
    text=[f"{score:.4f}" for score in importance_scores],
    textposition='outside',
    textfont=dict(size=12)
))

# Update layout
fig.update_layout(
    title="Top AI Factor Importance",
    xaxis_title="Feature Score",
    yaxis_title="Clinical Params",
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Reverse the y-axis order to show highest importance at the top
fig.update_yaxes(autorange="reversed")

# Save the chart
fig.write_image("clinical_ai_factors_chart.png")