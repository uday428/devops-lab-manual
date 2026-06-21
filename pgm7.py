import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# Linear Regression for California Housing Dataset
# =====================================================================
def linear_regression_california():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data[["AveRooms"]]  # Using only AveRooms as feature
    y = housing.target  # Median value of homes as target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Plotting results
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="blue", label="Actual", alpha=0.5)
    plt.plot(X_test, y_pred, color="red", label="Predicted", linewidth=2)
    plt.xlabel("Average number of rooms (AveRooms)")
    plt.ylabel("Median value of homes ($100,000)")
    plt.title("Linear Regression - California Housing Dataset")
    plt.legend()
    plt.show()
    
    # Print performance metrics
    print("=" * 45)
    print("Linear Regression - California Housing Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))
    print("=" * 45)

# =====================================================================
# Polynomial Regression for Auto MPG Dataset
# =====================================================================
def polynomial_regression_auto_mpg():
    # Load the Auto MPG dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    
    data = pd.read_csv(url, sep=r'\s+', names=column_names, na_values="?")
    data = data.dropna()  # Drop rows with missing values
    
    X = data["displacement"].values.reshape(-1, 1)  # Feature: displacement
    y = data["mpg"].values  # Target: mpg
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Polynomial Regression model (degree 2)
    poly_model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
    poly_model.fit(X_train, y_train)
    
    # Sort the test data points for a smooth plotted line
    sort_idx = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sort_idx]
    y_test_sorted = y_test[sort_idx]
    y_pred_sorted = poly_model.predict(X_test_sorted)
    
    # Plotting results
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test_sorted, y_test_sorted, color="blue", label="Actual", alpha=0.7)
    plt.plot(X_test_sorted, y_pred_sorted, color="red", label="Predicted (Degree 2)", linewidth=2)
    plt.xlabel("Displacement")
    plt.ylabel("Miles per gallon (mpg)")
    plt.title("Polynomial Regression - Auto MPG Dataset")
    plt.legend()
    plt.show()
    
    # Print performance metrics
    y_pred_all = poly_model.predict(X_test)
    print("\n" + "=" * 45)
    print("Polynomial Regression - Auto MPG Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_all))
    print("R^2 Score:", r2_score(y_test, y_pred_all))
    print("=" * 45)

# =====================================================================
# Main Execution Blocks
# =====================================================================
if __name__ == "__main__":
    linear_regression_california()
    polynomial_regression_auto_mpg()