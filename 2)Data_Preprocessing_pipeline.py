import numpy as np # Handles the calculations
import pandas as pd # Frame the data
from sklearn.impute import SimpleImputer # Fills missing values (Imputation)
from sklearn.preprocessing import OneHotEncoder, StandardScaler # One Encodes Texts and the other do the scaling

"""
NumPy (Math Engine): Correct. It handles the low-level, high-speed array math, matrices, and linear algebra behind the scenes.
Pandas (Data Table Manager): Correct. It gives us DataFrames (like Excel spreadsheets in Python)
 so we (humans) can easily load, organize, filter, and inspect row-and-column data.
Scikit-Learn (Machine Learning Toolkit): It handles preprocessing (imputation, encoding, scaling),
 splits the data, fits algorithms to training data, makes predictions, and tests performance metrics.
"""

"Here's the Messy Data"
# have missing values (np.nan)
# Has text categories ("Dhaka", "Chitagong", etc...)
# Masively different scales (Bedrooms vs Price/Sqft)

data = pd.DataFrame({
    "City": ["Dhaka", "Chitagong", np.nan, "Khulna", "Dhaka", "Sylhet"],
    "Bedrooms": [2, np.nan, 4, 3, 6, np.nan],
    "Square Feet": [1000, 1500, 2400, 1800, 1600, 3500]
})

print("The Messy data")
print(data)

# Time to handle the missing values

# Fill missing numbers with the average (mean) of the column
Number_Imputer = SimpleImputer(strategy='mean')

# Fit & Transform -- Learn the mean of Bedrooms then fill the missing entries
data[["Bedrooms"]] = Number_Imputer.fit_transform(data[["Bedrooms"]])

# For texts -- Fill missing city with most frequent city ("mode")
Category_Imputer = SimpleImputer(strategy='most_frequent')
data[["City"]] = Category_Imputer.fit_transform(data[["City"]])

# Now, time to Encode the Texts
# Initialize encoder (sparse_output = False -> gives us a readable array)
encoder = OneHotEncoder(sparse_output=False)

# Convert 'City' column into binary 1s and 0s
encoded_cities = encoder.fit_transform(data[["City"]])
city_cols = encoder.get_feature_names_out(["City"])

# Create a DataFrame for the new city columns
encoded_cities_df = pd.DataFrame(encoded_cities, columns=city_cols)

# Time for Feature Scaling (Standardization)
# Rescale numeric features so they share a common scale (mean = 0, std = 1)
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(data[["Bedrooms", "Square Feet"]])

# Create a DataFrame of the new scaled numbers
scaled_nums_df = pd.DataFrame(scaled_nums, columns=["Bedrooms", "Square Feet"])

# Time to combine all into clean Feature Matrix (M)

M_clean = pd.concat([encoded_cities_df, scaled_nums_df], axis=1)

print("\nClean Processed Data")
print(M_clean)