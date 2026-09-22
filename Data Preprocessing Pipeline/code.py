import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Creating the messy data
data = pd.DataFrame({
    "Area": ["Uttara", "Gulshan", np.nan, "Badda", np.nan, "Gulshan", "Banani"],
    "Item purchased": [np.nan, 3, 5, np.nan, 1, 5, 2],
    "Total Cost (BDT)": [1800, np.nan, 7000, 600, 9500, 11500, 2200]
})

number_imputer = SimpleImputer(strategy="mean") 
category_imputer = SimpleImputer(strategy="most_frequent") # Area as category

# Applying it on the data using fit_transform
data[["Area"]] = category_imputer.fit_transform(data[["Area"]])
data[["Item purchased"]] = number_imputer.fit_transform(data[["Item purchased"]])
data[["Total Cost (BDT)"]] = number_imputer.fit_transform(data[["Total Cost (BDT)"]])

# Encode & Scale
encoder = OneHotEncoder(sparse_output=False)

encode_category = encoder.fit_transform(data[["Area"]])
new_column = encoder.get_feature_names_out(["Area"]) # makes the new column

# Reframe Data
new_df_1 = pd.DataFrame(encode_category, columns=new_column)

# Scale
scaler = StandardScaler()

num_scaler = scaler.fit_transform(data[["Item purchased", "Total Cost (BDT)"]])

# Reframe again
new_df_2 = pd.DataFrame(num_scaler, columns=["Item purchased",  "Total Cost (BDT)"])

# Assemble the data using concat
encoded_form = pd.concat([new_df_1, new_df_2], axis=1)

print(encoded_form)

print("\n\n")
# Decode
decode_new_df_1 = pd.DataFrame(encoder.inverse_transform(new_df_1), columns=["Area"])
decode_new_df_2 = pd.DataFrame(scaler.inverse_transform(new_df_2), columns=["Item purchased", "Total Cost (BDT)"])

decoded_form = pd.concat([decode_new_df_1, decode_new_df_2], axis=1)
print(decoded_form)
