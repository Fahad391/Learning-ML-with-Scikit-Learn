Software Developer Salary Predictor
 A model that predicts a software developer's Monthly Salary in BDT based on two features: Years of Experience and Projects Completed.

 Steps:
 1) import libraries
 --Numpy for doing the math
 -- Linear Regression to predict salary
 -- train_test_split to split data into 80/20 [80% data to train the model and 20% data to test it's ability to predict]
 -- mean_square_error to measure the error to evaluate the model's ability to predict

2) Create Data and Divide into -- X = Features [Years of Experience and Projects Completed]
                -- Y = Target [Salary]

3) Split into train-test:
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = None) -- 0.2 = 20% and random_state = None will create unpredictable shuffeling and will contain different rows on different runtime

4) Train & Evaluate -- .fit() to train and .predict() to see the result

5) calculate error 

6) try on new data