import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
# 1. Load the California Housing Dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedianHouseValue'] = housing.target # Add target variable
# --- 2. Analyze Distributions (Histograms) ---
numerical_features = df.select_dtypes(include=['float64']).columns
print("--- Feature Distributions (Histograms) ---")
df[numerical_features].hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Features", y=1.02)
plt.tight_layout()
plt.show()
# --- 3. Detect Outliers (Box Plots & IQR) ---
print("\n--- Outlier Detection (Box Plots & IQR) ---")
num_plots = len(numerical_features)
fig, axes = plt.subplots(nrows=(num_plots + 1) // 2, ncols=2, figsize=(14, num_plots * 3))
axes = axes.flatten()
outliers_info = {}
for i, feature in enumerate(numerical_features):
# Box Plot
    sns.boxplot(x=df[feature], ax=axes[i])
axes[i].set_title(f'Box Plot of {feature}')
# IQR Method for Outlier Calculation
Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
outliers_info[feature] = len(outliers)
# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.suptitle("Box Plots for Outlier Detection", y=1.02)
plt.tight_layout()
plt.show()
# --- 4. Summary of Outliers ---
print("\n--- Summary of Outliers (IQR Method) ---")
for feature, count in outliers_info.items():
    print(f"{feature}: {count} outliers")
