import json # Parses raw incoming JSON web records
import pandas as pd # Structures records into tabular dataframes
import joblib # Loads pre-trained scaler and model artifacts from disk

# Step-1: Load Saved Artifacts from Disk
scaler = joblib.load('scaler.joblib')
model = joblib.load('conversion_model.joblib')

# Step-2: Parse Incoming Raw Production Data
with open('new_customers.json', 'r') as file:
    raw_data = json.load(file)

df_new_data = pd.DataFrame(raw_data)

# Separate identifier column from numerical feature inputs
customer_id = df_new_data['customer_id']
feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
new_features = df_new_data[feature_columns]

# Step-3: Preprocess New Data Using Saved Scaler
# Use ONLY .transform() — never fit_transform() on production inference data
new_features_scaled = scaler.transform(new_features)

# Step-4: Generate Predictions and Class Probabilities
predictions = model.predict(new_features_scaled)
probabilities = model.predict_proba(new_features_scaled)[:, 1] # Index 1 isolates Class 1 probability

# Step-5: Format and Output Deployment Dashboard
df_new_data['Predicted_Status'] = ['Purchased' if p == 1 else 'No Purchase' for p in predictions]
df_new_data['Conversion_Probability'] = (probabilities * 100).round(2)

output_columns = ['customer_id', 'Predicted_Status', 'Conversion_Probability']

print("\n--- Live Production Predictions ---")
print(df_new_data[output_columns].to_string(index=False))