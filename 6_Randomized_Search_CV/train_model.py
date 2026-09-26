from sklearn.datasets import make_classification # Generates synthetic e-commerce session data
from sklearn.model_selection import train_test_split # Splits dataset into training and test sets
from sklearn.preprocessing import StandardScaler # Normalizes numerical features for Logistic Regression
from sklearn.linear_model import LogisticRegression # The classifier algorithm being tuned
from sklearn.model_selection import RandomizedSearchCV # Randomly samples parameter settings with CV
from sklearn.metrics import accuracy_score, classification_report # Evaluates model performance on test data
import joblib # Saves and loads Python objects (scalers and models) to disk

# Step-1: CREATE SYNTHETIC CONVERSION DATASET
"Simulate 1000 site visitorss, each with 5 numeric feature columns"
x,y = make_classification(
    n_samples=1000, # Generate 1000 rows/examples
    n_features=5, # Each row has 5 input features/columns
    n_classes=2, # There are 2 possible classes, usually 0 and 1
    random_state=42 # Makes the random dataset reproducible
)

# Step-2: Train-Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Step-3: Preprocess Features (Feature Scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) # Compute mean/std and scale train data
x_test_scaled = scaler.transform(x_test)        # Apply SAME scaling rules to test data

# Step-4: Define Base Model and Expanded Parameter Distributions
"(1)Instantiate the raw classifier algorithm"
base_model = LogisticRegression(random_state=42)

"""(2)Construct an expanded hyperparameter dictionary using standard python lists
Total potential combinations = 10 (C options) * 2 (penalties) * 2 (solvers) = 40 combinations
"""

parameter_distributions = {
    'C': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Step-5: Setup Randomized Search CV
"""
(1) n_iter=8 tells the search engine: "Pick only 8 random configurations out of the 40!"
(2) Total Fits = 8 sampled combinations * 5 folds CV = 40 total model fits
(3) (Grid Search would have taken 40 * 5 = 200 fits!)
"""

random_search = RandomizedSearchCV(
    estimator=base_model, # Algorithm to tune
    param_distributions=parameter_distributions, # Menu of options to sample from
    n_iter=8,  # Budget dial: total random samples to evaluate
    cv=5, # 5-Fold Cross-Validation on training data
    scoring='accuracy', # Metric used to rank candidate combinations
    random_state=42, # Fixes seed so random sampling is reproducible
    verbose=1  # Prints progress logs while running
)

# Step-6: Execute Randomized Search On Training Data
"(1) Evaluate 8 random combinators across 5 folds behind the scenes"
random_search.fit(x_train_scaled, y_train)

# Step-7: Inspect Winning Results
print("Best Hyperparameters Found:", random_search.best_params_)
print("Best Average CV Score:", random_search.best_score_)

# Step-8: Evaluate Champion Model on Locked-Out Test Data
"(1) Refits the winning configuration on the full training set"
best_model = random_search.best_estimator_

"(2) Generate final predictions on unseen test data"
y_predict = best_model.predict(x_test_scaled)

print("\n--- Unseen Test Set Performance ---")
print("Test Accuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))

# Step-9: Save Artifacts to Disk for Deployment
# Save scaling rules (mean and std calculated from training data)
joblib.dump(scaler, 'scaler.joblib') 

# Save winning Randomized Search champion model weights
joblib.dump(best_model, 'conversion_model.joblib') 

print("\nScaler rules and Champion Model saved successfully to disk!")