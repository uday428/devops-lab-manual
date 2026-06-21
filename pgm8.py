import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Create Decision Tree Model
clf = DecisionTreeClassifier(random_state=42)
# Train Model
clf.fit(X_train, y_train)
# Predict Test Data
y_pred = clf.predict(X_test)
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Predict New Sample
new_sample = np.array([X_test[0]])
prediction = clf.predict(new_sample)
prediction_class = "Benign" if prediction[0] == 1 else "Malignant"
print(f"Predicted Class for the new sample: {prediction_class}")
# Plot Decision Tree
plt.figure(figsize=(12,8))
tree.plot_tree(
clf,
filled=True,
    feature_names=data.feature_names,
class_names=data.target_names
)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()