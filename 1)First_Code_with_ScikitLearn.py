"Based on What I learned and got on practice"
"I built a model that predicts House Prices based on two features: size (in square feet) and number of bedrooms."

# Import Necessary Libraries
import numpy as np # -> Handles fast mathematical operations under the hood.

# Now the main Machine Learning library Scikit-Learn also called sklearn
from sklearn.linear_model import LinearRegression #-> For making numerical predictions
from sklearn.model_selection import train_test_split #-> Splits data into training and testing sets | In simple Splitting data
from sklearn.metrics import mean_squared_error #-> Measures how much the predictions differ from actual values or we can say For measuring prediction error

# Time to Create Data (Features & Target)

# Features (X): [Square Feet, Bedrooms]
X = np.array([
    [600,1],
    [800,2],
    [1200,2],
    [1500,2],
    [1800,3],
    [2200,4],
    [2500,4]
])

# Target (Y): Price per sq feet in BDT (just assume)
Y = np.array([1250, 1500, 1800, 2260, 2350, 2500, 2700])

# Spliting Data into Tain & Test sets
"Split 80% data to teach the model, 20% to test it"

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# Train the Model using Linear Regression

"What Linear Regression does? -> Linear Regression draws a straight line through points"

model = LinearRegression()

# Train the algorithm using the training data
model.fit(X_train, Y_train)

# Time to make predictions

Predict_the_price_per_sQ = model.predict(X_test)

# Calculate error to see how far off the predictions were from actual test prices
error = np.sqrt(mean_squared_error(Y_test, Predict_the_price_per_sQ))

# The new data -> predict the price of these 2 houses

new_houses = [[1600,3], [2800,4]]
predict_it = model.predict(new_houses)

# Let's evaluate via the 20%, the test part to see what it does
for i in range(len(X_test)):
    print(f"House {X_test[i]} predicted {Predict_the_price_per_sQ[i]:.2f} BDT per SQ feet where Actual is {Y_test[i]:.2f} BDT")

# check the error
print(f"\nAverage Model Error (RMSE): ±{error:.2f} BDT")

# Time to see what it predicts on new data
print(f"Predicted Price for House 1 (1600 sqft, 3 bed): {predict_it[0]:.2f} BDT per SQfeet")
print(f"Predicted Price for House 2 (2800 sqft, 4 bed): {predict_it[1]:.2f} BDT per SQfeet")

