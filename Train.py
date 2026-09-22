import pandas as pd # For Data Framing
import matplotlib.pyplot as plt # For 2D Visualization

# Importing necessary Scikit-Learn models for scaling, clustering and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib  # For Saving Scikit-Learn models

# 1. Load data
DF = pd.read_csv('customer_data.csv')
# We select ONLY the features we want to cluster on: 'AnnualSpend' and 'FrequencyScore'.
# Note: We exclude 'CustomerID' because arbitrary IDs have no numerical meaning for distance calculations.
X = DF[['AnnualSpend', 'FrequencyScore']]

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
DF['Cluster'] = kmeans.fit_predict(X_scaled)

# Centers
scaled_centers = kmeans.cluster_centers_
original_centers = scaler.inverse_transform(scaled_centers)

# 4. Calculate metrics & summary data
score = silhouette_score(X_scaled, DF['Cluster'])
summary = DF.groupby('Cluster')[['AnnualSpend', 'FrequencyScore']].mean()
# Using Matplotlib for Visualization

# 1) Create figure with extra width on the right to make room for the text panel
fig, ax = plt.subplots(figsize=(11, 6))

# 2) Adjust plot area so it leaves 35% Margin space on the right
plt.subplots_adjust(right=0.65)

# 3) Scatter plot of customer clusters
scatter = ax.scatter(
    DF['AnnualSpend'],
    DF['FrequencyScore'],
    c=DF['Cluster'],
    cmap='viridis', 
    s=100,
    edgecolors='k'
)

# 4) plot Centroids
ax.scatter(
    original_centers[:, 0],
    original_centers[:, 1],
    s=250,
    c='red',
    marker='X',
    label='Centroids'
)

ax.set_title('Customer Segmentation via K-Means', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Annual Spend ($)')
ax.set_ylabel('Shopping Frequency Score (1-100)')
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

# TO DISPLAY SCORES & INSIGHTS ON THE FIGURE
# USING plt.figtext()

# 1. Title / Header for the Insights Box
plt.figtext(
    x=0.68, y=0.82, 
    s="Model Performance", 
    fontsize=12, fontweight='bold', color='#1f77b4'
)

# 2. Display Silhouette Score
plt.figtext(
    x=0.68, y=0.76, 
    s=f"Silhouette Score: {score:.3f}\n(Scale: -1 to +1 | >0.5 is strong)", 
    fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#1f77b4")
)

# 3. Header for Cluster Summary
plt.figtext(
    x=0.68, y=0.62, 
    s="Mean Stats Per Segment", 
    fontsize=12, fontweight='bold', color='#2ca02c'
)

# 4. Format the summary DataFrame into a readable string
summary_text = "Cluster  | Spend ($) | Frequency\n"
summary_text += "-" * 32 + "\n"
for cluster_id, row in summary.iterrows():
    summary_text += f"   {cluster_id}     |  ${row['AnnualSpend']:>6.0f}  |   {row['FrequencyScore']:>5.1f}\n"

# Render the formatted summary table using a monospaced font
plt.figtext(
    x=0.68, y=0.38, 
    s=summary_text, 
    fontsize=9.5, family='monospace',
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f4f4", edgecolor="#cccccc")
)

plt.show()

joblib.dump(scaler, 'scaler.joblib')
joblib.dump(kmeans, 'kmeans_model.joblib')