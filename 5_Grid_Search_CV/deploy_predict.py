import json # Used for parsing incoming raw JSON customer files
import pandas as pd # Used to structure data into column format
import joblib # Used to load saved scaler and model artifacts

# Step-1: Load Saved artifact from disk
"Load the pre-trained scaler (retains mean and variance settings from training)"
scaler = joblib.load('scaler.joblib')

"Load the winning Grid Search model (retains tuned weights and hyperparameters)"
model = joblib.load('churn_model.joblib')

# Step-2: Read and Parse Incoming production Data
"Use open and read JSON records from production system"
with open('new_customers.json', 'r') as file:
    raw_data = json.load(file)

# Convert list of JSON objects into a Pandas DataFrame
df_new_data = pd.DataFrame(raw_data)

# Separate identifier metadata from the 5 numerical features used during training
customer_id = df_new_data['customer_id']
feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
new_feature = df_new_data[feature_columns]

# Step-3: Preprocess new Data using Saved Scaler
"Use ONLY .transform() — never fit_transform() on production data"
new_feature_scaled = scaler.transform(new_feature)

# Step-4: Generate Predictions and Probabilities
"Predict discrete class labels (0 = Retain, 1 = Churn)"
predictions = model.predict(new_feature_scaled)

"Calculate exact confidence probabilites (percentage chance of churn)"
probabilities = model.predict_proba(new_feature_scaled)[:, 1] # Probability of Class 1 (Churn)

# Step-5: Format and Output Deployment Results
"Attach prediction outcomes back to customer IDs for business action"
df_new_data['Predicted_Status'] = ['Churn' if p == 1 else 'Retain' for p in predictions]
df_new_data['Churn_Probability'] = (probabilities * 100).round(2)

# Print clean dashboard output for operational usage
output_columns = ['customer_id', 'Predicted_Status', 'Churn_Probability']
print("\n--- Live Production Predictions ---")
print(df_new_data[output_columns].to_string(index=False))