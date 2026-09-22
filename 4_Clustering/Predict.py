import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Explicit imports for clarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================================
# 1. LOAD SAVED MODEL & SCALER
# =====================================================================
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans_model.joblib')

# =====================================================================
# 2. LOAD UNSEEN CUSTOMER DATA
# =====================================================================
DF = pd.read_csv('unseen_customer_data.csv')
Feature = DF[['AnnualSpend', 'FrequencyScore']]

# =====================================================================
# 3. TRANSFORM & PREDICT (UNDER-THE-HOOD MATH)
# =====================================================================
# Step A: Transform using saved mean and std_dev (Z-score calculation)
Feature_scaled = scaler.transform(Feature)

# Step B: Calculate distance to nearest frozen centroid
DF['Assigned_Cluster'] = kmeans.predict(Feature_scaled)

# Recover original centroid coordinates for visualization
scaled_centers = kmeans.cluster_centers_
original_centers = scaler.inverse_transform(scaled_centers)


# =====================================================================
# 4. VISUALIZATION WITH MATPLOTLIB & FIGTEXT (SIDE PANEL)
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 6))

# Reserve 35% margin space on the right for math/stats display
plt.subplots_adjust(right=0.65)

# Plot unseen customers (large stars colored by predicted cluster)
scatter = ax.scatter(
    DF['AnnualSpend'],
    DF['FrequencyScore'],
    c=DF['Assigned_Cluster'],
    cmap='viridis', 
    s=200,
    marker='*',
    edgecolors='k',
    label='New Customers'
)

# Plot Centroids
ax.scatter(
    original_centers[:, 0],
    original_centers[:, 1],
    s=250,
    c='red',
    marker='X',
    label='Centroids'
)

ax.set_title('Live Inferences: Unseen Customer Assignments', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Annual Spend ($)')
ax.set_ylabel('Shopping Frequency Score (1-100)')
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)


# =====================================================================
# 5. DISPLAY MATHEMATICAL CALCULATIONS ON THE FIGURE USING FIGTEXT
# =====================================================================

# 1. Header
plt.figtext(
    x=0.68, y=0.82, 
    s="Model Scaler Parameters", 
    fontsize=12, fontweight='bold', color='#1f77b4'
)

# 2. Display Saved Mean & Variance used for Transformation
mean_spend, mean_freq = scaler.mean_[0], scaler.mean_[1]
std_spend, std_freq = scaler.scale_[0], scaler.scale_[1]

scaler_stats = f"Mean Spend: ${mean_spend:.0f} | Std: ${std_spend:.0f}\n"
scaler_stats += f"Mean Freq:   {mean_freq:.1f} | Std:  {std_freq:.1f}\n"
scaler_stats += "(Formula: Z = (X - Mean) / Std)"

plt.figtext(
    x=0.68, y=0.70, 
    s=scaler_stats, 
    fontsize=9.5, family='monospace',
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#1f77b4")
)

# 3. Header for Inferred Predictions
plt.figtext(
    x=0.68, y=0.58, 
    s="Unseen Data Math Calculations", 
    fontsize=12, fontweight='bold', color='#2ca02c'
)

# 4. Build text showing Raw -> Scaled (Z) -> Cluster
calc_text = "ID  | Raw ($ / Freq) | Scaled (Z_s / Z_f) | Cluster\n"
calc_text += "-" * 53 + "\n"

for i, row in DF.iterrows():
    cid = int(row['CustomerID'])
    spend, freq = row['AnnualSpend'], row['FrequencyScore']
    z_spend, z_freq = Feature_scaled[i][0], Feature_scaled[i][1]
    cluster = row['Assigned_Cluster']
    
    calc_text += f"{cid} | ${spend:>5.0f} / {freq:>2.0f}     | {z_spend:>+5.2f} / {z_freq:>+5.2f}     |    {cluster}\n"

# Render the formatted calculations
plt.figtext(
    x=0.68, y=0.28, 
    s=calc_text, 
    fontsize=8.5, family='monospace',
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f4f4", edgecolor="#cccccc")
)

plt.show()