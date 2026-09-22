import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load Data
laptop_data = pd.read_csv("laptops.csv")

# Feature-Target Split
Feature = laptop_data[["Brand", "RAM_GB", "Storage_GB"]] # 2D 
Target = laptop_data["Price_BDT"] # 1D

# Train-Test Split
Feature_Train, Feature_Test, Target_Train, Target_Test = train_test_split(Feature, Target, test_size=0.2)

# Imputation
number_imputer = SimpleImputer(strategy="mean")
category_imputer = SimpleImputer(strategy="most_frequent")

# Use fit_transform() for train and transform() for test
train_num = number_imputer.fit_transform(Feature_Train[["RAM_GB", "Storage_GB"]])
train_category = category_imputer.fit_transform(Feature_Train[["Brand"]])

test_num = number_imputer.transform(Feature_Test[["RAM_GB", "Storage_GB"]])
test_category = category_imputer.transform(Feature_Test[["Brand"]])

# Feature Transform Encode Category & Scale Numericals
encode = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
Scaler = StandardScaler()

# Encode & Scale Training Data
encode_category_train =  encode.fit_transform(train_category)
Scale_numericals_train = Scaler.fit_transform(train_num)


# Assemble Clean Training Matrix
clean_train_data = np.hstack([encode_category_train, Scale_numericals_train])

# Encode & Scale Testing Data
encode_category_test = encode.transform(test_category)
Scale_numericals_test = Scaler.transform(test_num)

# Assemble Clean Testing Matrix
clean_test_data = np.hstack([encode_category_test, Scale_numericals_test])

# Train the Model
Model = LinearRegression()

Model.fit(clean_train_data, Target_Train)

# Test it
predict_price = Model.predict(clean_test_data)

# Validate the modek
error_measure = np.sqrt(mean_squared_error(Target_Test, predict_price))

# Reset target test indices so we can iterate smoothly side-by-side
actual_prices = Target_Test.values

for i in range(len(predict_price)):
    print(
        f"Sample {i+1} predicted {predict_price[i]:.2f} BDT | "
        f"Actual price is {actual_prices[i]:.2f} BDT"
    )

print(f"Error: {error_measure:.2f} BDT")