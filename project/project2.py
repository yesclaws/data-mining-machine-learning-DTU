import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
from sklearn.linear_model import LinearRegression

# Regression Part A

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Load the data and preprocess as provided in your code
# Load the data
df = pd.read_csv('/Users/shanelim/Desktop/Sem 2.1/Exchange/DTU Courses/Intro to Data Mining/Project 2/data.csv')

columns_to_drop = ['Unnamed: 32', 'id']
df = df.drop(columns=columns_to_drop)

print("Malignant = 1, Benign = 0")
df["diagnosis"]= df["diagnosis"].map(lambda row: 1 if row=='M' else 0)

# Selected features
y = df['radius_mean']
X = df[['concave points_mean', 'concavity_mean', 'concave points_mean', 'perimeter_mean', 
        'area_mean', 'compactness_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'concavity_worst', 
        'concave points_worst', 'fractal_dimension_mean']]

selected_attributes = X.values
print(selected_attributes)

scaled_X = StandardScaler().fit_transform(X)

#print(scaled_X)
#print(scaled_X.shape)

# Regression Part B

optimal_h = []
optimal_lambda = []

#Apply regularization
# Split the Data into Training and Testing Sets:
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

# Initialize the Ridge Regression model with an alpha value
ridge_model = Ridge(alpha=1)

# Fit the model on the training data
ridge_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = ridge_model.predict(X_test)

# Calculating the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Ridge Regression setup
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
print("Mean Squared Error (Ridge):", mean_squared_error(y_test, y_pred))

# Initial exploratory phase for ANN
hidden_layer_sizes = [(1,), (5,), (10,), (50,), (100,)]
alphas = [0.0001, 0.001, 0.01, 0.1, 1]

# This will store the best score and corresponding hyperparameters
best_score = np.inf
best_params = {}

for h in hidden_layer_sizes:
    for alpha in alphas:
        ann_model = MLPRegressor(hidden_layer_sizes=h, alpha=alpha, max_iter=1000, random_state=1)
        scores = cross_val_score(ann_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mean_mse = -np.mean(scores)
        print(f"Hidden Layers: {h}, Alpha: {alpha}, Mean MSE: {mean_mse}")
        
        if mean_mse < best_score:
            best_score = mean_mse
            best_params = {'hidden_layer_sizes': h, 'alpha': alpha}

print(f"Best parameters found: {best_params} with Mean MSE: {best_score}")

# Use the best parameters found in the exploratory phase for detailed GridSearchCV
ann_params = {
    'hidden_layer_sizes': [best_params['hidden_layer_sizes']],  # Refine based on the exploratory phase
    'alpha': [best_params['alpha']],  # Refine based on the exploratory phase
    'max_iter': [50],  # Reduced max_iter
    'early_stopping': [True],  # Enable early stopping
}

# Use the best parameters found in the exploratory phase
best_ann_model = MLPRegressor(hidden_layer_sizes=best_params['hidden_layer_sizes'], 
                              alpha=best_params['alpha'], 
                              max_iter=1000,  # or any other suitable number of iterations
                              random_state=1)

# Define the hyperparameter grid for Ridge
ridge_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # Example range of alpha values
}

# Perform two-level cross-validation with GridSearchCV inside the outer CV loop
K1 = K2 = 5
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=K2, shuffle=True, random_state=1)

# Store errors for each model
ridge_errors = []
dummy_errors = []
ann_errors = []
# Reset the list to store errors for the best ANN model
best_ann_errors = []

for train_index, test_index in outer_cv.split(X_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Train the best ANN model on the current outer training fold
    best_ann_model.fit(X_train_fold, y_train_fold)
    
    # Predict and calculate mean squared error for the best ANN model on the current outer test fold
    mse_best_ann = mean_squared_error(y_test_fold, best_ann_model.predict(X_test_fold))
    
    best_ann_errors.append(mse_best_ann)

# Now, best_ann_errors contains the MSEs of the best ANN model across all outer folds
mean_mse_best_ann = np.mean(best_ann_errors)

# Make sure y_train is a pandas Series, so .iloc will work
y_train = pd.Series(y_train)

for train_index, test_index in outer_cv.split(X_train):
    # Make sure to use .iloc when indexing pandas objects
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inner CV for hyperparameter tuning of Ridge
    ridge_search = GridSearchCV(Ridge(), ridge_params, cv=inner_cv)
    ridge_search.fit(X_train_fold, y_train_fold)
    ridge_model = ridge_search.best_estimator_
    optimal_lambda.append(ridge_search.best_params_['alpha'])  # Store the best lambda

    # Inner CV for hyperparameter tuning of ANN
    ann_search = GridSearchCV(MLPRegressor(random_state=1), ann_params, cv=inner_cv)
    ann_search.fit(X_train_fold, y_train_fold)
    ann_model = ann_search.best_estimator_
    optimal_h.append(ann_search.best_params_['hidden_layer_sizes'][0]) 

    # Baseline model
    dummy_model = DummyRegressor(strategy="mean").fit(X_train_fold, y_train_fold)

    # Predict and calculate mean squared error for each model
    mse_ridge = mean_squared_error(y_test_fold, ridge_model.predict(X_test_fold))
    mse_ann = mean_squared_error(y_test_fold, ann_model.predict(X_test_fold))
    mse_dummy = mean_squared_error(y_test_fold, dummy_model.predict(X_test_fold))

    ridge_errors.append(mse_ridge)
    ann_errors.append(mse_ann)
    dummy_errors.append(mse_dummy)

# Calculate mean errors across folds
mean_mse_ridge = np.mean(ridge_errors)
mean_mse_ann = np.mean(ann_errors)
mean_mse_dummy = np.mean(dummy_errors)

# Statistical comparison
t_stat, p_val_ridge_dummy = ttest_rel(ridge_errors, dummy_errors)
t_stat, p_val_ann_dummy = ttest_rel(ann_errors, dummy_errors)
t_stat, p_val_ridge_ann = ttest_rel(ridge_errors, ann_errors)

# Output the results
print("Mean Squared Error (Ridge):", mean_mse_ridge)
print("Mean Squared Error (ANN):", mean_mse_ann)
print("Mean Squared Error (Dummy):", mean_mse_dummy)
print()
print("P-Value (Ridge vs Dummy):", p_val_ridge_dummy)
print("P-Value (ANN vs Dummy):", p_val_ann_dummy)
print("P-Value (Ridge vs ANN):", p_val_ridge_ann)

# Create a DataFrame
df_results = pd.DataFrame({
    'Outer fold': range(1, K1 + 1),  # or simply list(range(1, 11)) if K1 is 10
    'ANN h*': optimal_h,
    'ANN Etest': ann_errors,
    'Linear regression lambda*': optimal_lambda,
    'Linear regression Etest': ridge_errors,
    'Baseline Etest': dummy_errors
})

# Display the DataFrame as a table
print(df_results)

latex_table = df_results.to_latex(index=False, caption='Two-level cross-validation table used to compare the three models', label='tab:crossval')
print(latex_table)

from scipy import stats

# Function to calculate confidence intervals
def compute_confidence_intervals(data1, data2, alpha=0.05):
    # Calculate the differences
    differences = np.array(data1) - np.array(data2)
    
    # Calculate the mean and standard deviation of the differences
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # Calculate the standard error of the differences
    se_diff = std_diff / np.sqrt(len(differences))
    
    # Degrees of freedom
    df = len(differences) - 1
    
    # t-critical value for 95% confidence
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Lower and upper bound of the confidence interval
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    return ci_lower, ci_upper

# Example usage
alpha = 0.05  # For a 95% confidence interval

# Compute confidence intervals for the differences between models
ci_ridge_ann = compute_confidence_intervals(ridge_errors, best_ann_errors, alpha)
ci_ann_dummy = compute_confidence_intervals(ann_errors, dummy_errors, alpha)
ci_ridge_dummy = compute_confidence_intervals(ridge_errors, dummy_errors, alpha)
ci_best_ann_dummy = compute_confidence_intervals(best_ann_errors, dummy_errors, alpha)

# Print the results
print(f"Confidence Interval (Ridge vs ANN): {ci_ridge_ann}")
print(f"Confidence Interval (ANN vs Dummy): {ci_ann_dummy}")
print(f"Confidence Interval (Ridge vs Dummy): {ci_ridge_dummy}")
print(f"Confidence Interval (Best ANN vs Dummy): {ci_best_ann_dummy}")

# Classification
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2
import scipy.stats
import scipy.stats as st

csv1 = pd.read_csv("breast_cancer_features.csv")
csv2 = pd.read_csv("breast_cancer_targets.csv")

combined_csv = pd.concat([csv1, csv2], axis=1)

combined_csv.to_csv("combined_breast_cancer.csv", index=False)

combined_csv

X = combined_csv.drop(columns=['Diagnosis'])
y = combined_csv['Diagnosis']


lambda_values_list = []
test_errors_list = []

# Fixed lambda value
optimal_lambda = 0

# Perform 2-fold cross-validation
kf_outer = KFold(n_splits=10, shuffle=True, random_state=42)  # Outer loop with 10 folds
for train_index, test_index in kf_outer.split(X):
    X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
    y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

    if optimal_lambda == 0:
        # If optimal lambda is 0, set C to a very large value to remove regularization
        C_val = 1e10
    else:
        C_val = 1/optimal_lambda

    # Train logistic regression model with fixed λ value
    final_model = LogisticRegression(C=C_val, solver='liblinear')  # C = 1/λ
    final_model.fit(X_train_outer, y_train_outer)

    # Evaluate model on outer test set
    test_error = 1 - accuracy_score(y_test_outer, final_model.predict(X_test_outer))  # Misclassification error

    # Store the chosen λ value for the outer fold
    lambda_values_list.append(optimal_lambda)

    # Store the test error for the current outer fold
    test_errors_list.append(test_error)

answer = 0

# Print λi and Ei_test for each fold
for i, (lambda_val, test_error) in enumerate(zip(lambda_values_list, test_errors_list), 1):
    answer = answer + test_error
    print(f"Outer Fold {i}: λ{i} = {lambda_val}, E{i}_test = {test_error}")

print(answer)

"""Baselin Ei values"""

majority_class = y.value_counts().idxmax()

# Initialize list to store test errors
test_errors = []

# Perform bootstrapping to compute test error
for i in range(10):
    # Generate bootstrap sample
    bootstrap_sample = np.random.choice(y, size=len(y), replace=True)

    # Predict every instance in the bootstrap sample to belong to the majority class
    baseline_predictions = [majority_class] * len(bootstrap_sample)

    # Compute the test error
    test_error = 1 - accuracy_score(bootstrap_sample, baseline_predictions)

    # Store the test error
    test_errors.append(test_error)

# Print the test errors for each iteration
for i, error in enumerate(test_errors, 1):
    print(f"E{i}_test:", error)

"""ANN :"""

test_errors_list = []

hidden_units = 3 #USE hi = 3

# Perform K-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the ANN model
    model = MLPClassifier(hidden_layer_sizes=(hidden_units,), activation='relu', solver='adam', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_pred)

    # Store the test error for the current fold
    test_errors_list.append(test_error)

answer = 0

# Print Ei_test for each fold
for i, test_error in enumerate(test_errors_list, 1):
    answer = answer + test_error
    print(f"E{i}_test = {test_error}")

print(answer)

"""# Statistical Evaluation"""

#y_hat_baseline

majority_class = y.value_counts().idxmax()

# Predict the majority class for all instances in the test data
y_hat_baseline = [majority_class] * len(y)

# Print the predictions
print(y_hat_baseline)
print(len(y_hat_baseline))

# Define and train the MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42)
mlp_classifier.fit(X, y)

# Predict labels for test data
y_hat_ann = mlp_classifier.predict(X)

# Print the predictions
print(y_hat_ann)
print(len(y_hat_ann))

# Define and train the logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Predict labels for test data
y_hat_logreg = logreg_model.predict(X)

# Print the predictions
print(y_hat_logreg)

"""Run McNemar's Test"""

y = np.array(y)

def mcnemarTest(y_true, yhatA, yhatB, alpha=0.05):
    """
    Perform McNemar's test to compare the accuracy of two classifiers.

    Parameters:
    - y_true: array-like, true labels
    - yhatA: array-like, predicted labels by classifier A
    - yhatB: array-like, predicted labels by classifier B
    - alpha: float, significance level (default: 0.05)

    Returns:
    - thetahat: float, estimated difference in accuracy between classifiers A and B
    - CI: tuple, confidence interval of the estimated difference in accuracy
    - p: float, p-value for the two-sided test of whether classifiers A and B have the same accuracy
    """

    nn = np.zeros((2, 2))
    c1 = yhatA == y_true
    c2 = yhatB == y_true

    nn[0, 0] = sum(c1 & c2)
    nn[0, 1] = sum(c1 & ~c2)
    nn[1, 0] = sum(~c1 & c2)
    nn[1, 1] = sum(~c1 & ~c2)

    n = sum(nn.flat)
    n12 = nn[0, 1]
    n21 = nn[1, 0]

    thetahat = (n12 - n21) / n
    Etheta = thetahat

    Q = (
        n**2
        * (n + 1)
        * (Etheta + 1)
        * (1 - Etheta)
        / ((n * (n12 + n21) - (n12 - n21) ** 2))
    )

    p = (Etheta + 1) * 0.5 * (Q - 1)
    q = (1 - Etheta) * 0.5 * (Q - 1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1 - alpha, a=p, b=q))

    p = 2 * scipy.stats.binom.cdf(min([n12, n21]), n=n12 + n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12 + n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=", (n12 + n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print(
        "p-value for two-sided test A and B have same accuracy (exact binomial test): p=",
        p,
    )

    return thetahat, CI, p

alpha = 0.05

[thetahat, CI, p] = mcnemarTest(y, y_hat_baseline, y_hat_logreg, alpha = alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

[thetahat, CI, p] = mcnemarTest(y, y_hat_baseline, y_hat_ann, alpha = alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

[thetahat, CI, p] = mcnemarTest(y, y_hat_logreg,y_hat_ann, alpha = alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

# Convert diagnosis to binary
df["diagnosis"] = df["diagnosis"].map(lambda x: 1 if x == 'M' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

# Logistic Regression with lambda = 0, which translates to no regularization (C = np.inf)
# Note: Setting C very high to simulate no regularization.
logistic_model = LogisticRegression(C=1e10, max_iter=1000)

# Fit the model on the training data
logistic_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = logistic_model.predict(X_test)

# Calculating the accuracy
accuracy = logistic_model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Displaying the coefficients (feature importance)
print("Coefficients:", logistic_model.coef_)

# Find coefficients for ridge model
# Initialize the Ridge Regression model with an alpha value
ridge_model = Ridge(alpha=1)

# Fit the model on the training data
ridge_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = ridge_model.predict(X_test)

# Calculating the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

ridge_model = LinearRegression()

# Fit the model on your training data
ridge_model.fit(X_train, y_train)  # Assuming X_train and y_train are already defined

# Retrieve the coefficients of the fitted model
linear_coefficients = ridge_model.coef_

# Display the coefficients
print("Coefficients of the Linear Regression model:", linear_coefficients)