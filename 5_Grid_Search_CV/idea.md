A telecommunications company wants to predict whether a subscriber will cancel their service (1 = Churn) or remain a customer (0 = Retain).

Using customer account data (e.g., tenure length, monthly charges, customer support tickets), build a LogisticRegression model and use GridSearchCV to systematically test combinations of regularization strengths (C) and penalty types (penalty) to find the configuration that gives the highest accuracy.

Train the model -> save it -> Deploy & Test it