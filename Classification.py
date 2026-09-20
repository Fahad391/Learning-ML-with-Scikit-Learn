"Predicting whether a customer buys a product (1 = Buy, 0 = No Buy) based on their Age and Estimated Salary."

# Import necessary libraries
import numpy as np # for numerical operations
import pandas as pd # for managing dataset
from sklearn.model_selection import train_test_split # split data into training-testing
from sklearn.preprocessing import StandardScaler # Preprocess/Scale features
from sklearn.linear_model import LogisticRegression # Train the Classification Model
from sklearn.metrics import accuracy_score # Evaluate predictions

# Step-1: Create a dataset
"""(Features: Age, Salary in BDT | Target: 1=Buy, 0=No Buy).
# Younger people with lower salaries generally didn't buy; older/higher salary people did."""

data = {
    "Age": [22, 25, 28, 30, 45, 50, 52, 55],
    "Salary (BDT)k": [17, 25, 30, 65, 85, 105, 120, 220], # with k it mean 17k, 25k,....
    "Bought": [0,0,0,0,1,1,1,1]
}

# Frame the data using pandas
DFrame = pd.DataFrame(data)

# Step-2: Feature-Target split
"Feature (X) = Independent variables (inputs) | Target(Y) = Dependent variable (target output)"

X =  DFrame[["Age", "Salary (BDT)k"]] # Feature
Y = DFrame["Bought"] # Target

# Step-3: Train-Test Split
"Train 60% & Test 40%"

X_train, X_test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.4, random_state=100)

# Step-4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn scaling parameters & transform training data | Here Fit trains and transform converts raw numbers into scaled numbers
X_test_scaled = scaler.transform(X_test) # Transform test data using same parameters

# Step-5: Train the Model
model = LogisticRegression() # Create Logistic Regression model
model.fit(X_train_scaled, Y_Train) # Model learns weights for Age and Salary under the hood

# Step-6: Predict the unseen Test Data
y_predict = model.predict(X_test_scaled) # Output predicted classes (0 or 1)
y_Probability = model.predict_proba(X_test_scaled)[:,1] # Output internal probability of class 1 (Buying)

# Step-7: Evaluate Performance
accuracy = accuracy_score(Y_Test, y_predict) # Calculate fraction of correct predictions

# Time to print the results
print("Result\n")
results_df = pd.DataFrame({
    "Actual Class": Y_Test.values,
    "Predicted Class": y_predict,
    "Buy Probability": np.round(y_Probability, 2)
})
print(results_df.to_string(index=False))
print("-------------------------------------")
print(f"Model Accuracy: {accuracy * 100:.1f}%")


# Step-8: Predict on New data 
new_data = pd.DataFrame({
    "Age": [40,18,35,26],
    "Salary (BDT)k": [75, 16, 220, 700]
})

# Important to Note: Always scale new data using the "Already Trained scaler"
new_data_scaled = scaler.transform(new_data)

# Make Prediction
predict_it = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)[:,1]
new_accuracy = accuracy_score(Y_Test,predict_it)

print("\nAfter Trying the model on New Data\n")
# using for loop to loop through Age, Salary, Prediction and Probability together
for age, salary, pred, prob in zip(new_data["Age"], new_data["Salary (BDT)k"], predict_it, probability):
    # Using conditional statement
    if pred == 1:
        print(f"Customer Age: {age}, Salary: {salary}k BDT -> will Buy it (Confidence: {prob * 100:.1f}%)")
    else:
        print(f"Customer  Age: {age}, Salary: {salary}k BDT -> will not Buy it (Confidence: {(1 - prob) * 100:.1f}%)")

print(f"Model Accuracy Now: {new_accuracy * 100:.1f}%")