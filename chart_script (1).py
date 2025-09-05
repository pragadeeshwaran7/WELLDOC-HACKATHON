import plotly.express as px
import plotly.graph_objects as go

# Data from the provided JSON
risk_categories = ["Very Low", "Low", "Moderate", "High", "Very High"]
patient_counts = [127, 210, 27, 18, 8]
percentages = [32.6, 53.8, 6.9, 4.6, 2.1]
colors = ["#28a745", "#6dbf47", "#ffc107", "#fd7e14", "#dc3545"]

# Create labels with both percentage and count (keeping under 15 char limit)
labels = [f"{cat}<br>{pct}% ({count})" for cat, pct, count in zip(risk_categories, percentages, patient_counts)]

# Create pie chart
fig = go.Figure(data=[go.Pie(
    labels=risk_categories,
    values=patient_counts,
    marker=dict(colors=colors),
    text=labels,
    textinfo="text",
    textposition="inside"
)])

# Update layout (title under 40 characters)
fig.update_layout(
    title="Patient Risk: 90-Day Deterioration",
    uniformtext_minsize=14,
    uniformtext_mode='hide'
)

# Save the chart
fig.write_image("patient_risk_distribution.png")