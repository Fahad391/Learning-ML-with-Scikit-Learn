import pandas as pd
import numpy as np
import Model_Training

# Access the trained model and preprocessors
model = Model_Training.Model
num_imputer = Model_Training.number_imputer
cat_imputer = Model_Training.category_imputer
encoder = Model_Training.encode
scale = Model_Training.Scaler

# load new data
new_data = pd.read_csv("new_data.csv")

#Impute missing values (using the rules learned from training)
new_num = num_imputer.transform(new_data[["RAM_GB", "Storage_GB"]])
new_cat = cat_imputer.transform(new_data[["Brand"]])

# Transfrom Encoding Categories & Scale Numericals
new_encoded = encoder.transform(new_cat)
new_scaled = scale.transform(new_num)

# Assemble clean data
new_clean_data = np.hstack([new_encoded, new_scaled])

# Make Predictions
Predict_prices = model.predict(new_clean_data)

print("New Laptop Price Prediction")
for i in range(len(Predict_prices)):
    brand = new_data["Brand"].iloc[i]
    ram = new_data["RAM_GB"].iloc[i]
    storage = new_data["Storage_GB"].iloc[i]
    print(f"Laptop {i+1} [{brand}, {ram}GB RAM, {storage}GB Storage] -> Predicted Price: {Predict_prices[i]:.2f} BDT")