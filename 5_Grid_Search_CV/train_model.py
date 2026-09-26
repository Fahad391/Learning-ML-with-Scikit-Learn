from sklearn.datasets import make_classification # Generates dummy data simulating our churn dataset
from sklearn.model_selection import train_test_split # Splits data into Training & Testing sets\
from sklearn.preprocessing import StandardScaler # Scales features (required for Logistic Regression to converge)
from sklearn.linear_model import LogisticRegression # The classification model whose settings are going to be tuning
from sklearn.model_selection import GridSearchCV # Automates testing hyperparameter combinations with CV
from sklearn.metrics import accuracy_score, classification_report # Evaluates final model performance on test data
import joblib # Library used to save and load ML models to disk

# Step-1 Create a Synthetic Churn Data-Set
# x is the input/features that ML model use to make predictions
# y is the target/label that the model tries to predict
x,y = make_classification(
    n_samples=1000, # Simulate 1000 customers
    n_features=5, # eachh with 5 numeric feature columns
    n_classes=2, # 2 possible classes (0 and 1)
    random_state=42 # Makes the random dataset reproducible
)

# Step-2 Split data into Train & Test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# step 3: Preprocess Features (Feature Scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) # Compute mean/std and scale train data
x_test_scaled = scaler.transform(x_test) # Apply SAME scaling rules to test data

# Step 4 Define Base Model and HyperParameter Grid

# 1. Instantiate the raw classifier algorithm
base_model = LogisticRegression(random_state=42)

# 2. Construct the hyperParameter "menu" (dictionary of lists)
# -C: Regularization strength (smaller - stronger penalty)
# -penalty: Regularization type ('l1' or 'l2')
# -solver: Engine that handles penalty calculation ('liblinear' supports l1 and l2)

parameter_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver':['liblinear']
}

# Step-5 Setup Grid Search CV

# Total Combinations = 5 (C values) * 2 (penalties) = 10 unique parameter sets
# Total Runs = 10 combinations * 5 folds CV = 50 total model fits
grid_search = GridSearchCV(
    estimator=base_model, # Algorithm to tune
    param_grid=parameter_grid, # The menu of options to test
    cv=5, # 5-Fold Cross-Validation on training data
    scoring='accuracy', # Metric used to rank the parameter combinations
    verbose=1 # Prints progress logs while running
)

# Step-6 Execute Grid Search on Training Data

# Fits all 50 variations behind the scenes and picks the top-performing setup
grid_search.fit(x_train_scaled, y_train)

# Step-7 Inspect wining Results
print("Best Hyperparameters Found:", grid_search.best_params_)
print("Best Average CV Score:", grid_search.best_score_)

# Step-8 Evaluate Champion Model on Locked-Out Test Data
best_model = grid_search.best_estimator_ # grid_search automatically refits the best model on the entire X_train set
y_predict = best_model.predict(x_test_scaled) # Generate final predictions on unseen test data

print("\n-- Unseen Test Set Performance ---")
print("Test Accuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))

# Save the model for deployment use

# Save the scaler rules (mean/std computed from training data)
joblib.dump(scaler, 'scaler.joblib') 

# Save the tuned champion model weights
joblib.dump(best_model, 'churn_model.joblib') 

print("Scaler and Champion Model saved successfully!")