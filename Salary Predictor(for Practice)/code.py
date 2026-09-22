import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create the data and divide into X & Y
# Features
X = np.array([  # [Years of Experience, Projects Completed]
    [1, 1],
    [2, 3],
    [6, 9],
    [3, 3],
    [2,4], 
    [10, 14], 
    [5, 7], 
    [15, 20]
])

# Target
Y = np.array([35000, 55000, 80000, 50000, 65000, 160000, 134000, 186000]) # BDT/Month

# Split into test & train [80/20]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= None)

# Time to use Linear Regression
model = LinearRegression() # called Linear Regression

# Time for .fit() and .predict() to do their work
model.fit(X_train, Y_train) # Trains on 80% data of feature & target

test_predict = model.predict(X_test)

# measure error
error = np.sqrt(mean_squared_error(Y_test, test_predict))

# using for loop and range(len(test_predict)) to test 20% part as index
for index in range(len(X_test)):
    print(f"For {X_test[index]} predicted {test_predict[index]:.1f} BDT/Month | Actual Value is {Y_test[index]:.1f} BDT/Month")

print(f"\nError Occured: {error:.1f} BDT")

# Try with new data
new_data = np.array([
    [4,11],
    [7, 16]
])

prediction = model.predict(new_data)
print('\n')

for i in range(len(new_data)):
    print(f"for {new_data[i]} salary is {prediction[i]:.1f} BDT/Month")

