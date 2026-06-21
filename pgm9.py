import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
# Fixed: Wrapped the multi-line import in parentheses to allow the line break
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix)

# Load Dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross Validation
cv_score = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation Accuracy: {cv_score.mean()*100:.2f}%")

# Display Test Images
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, true, pred in zip(axes.ravel(), X_test, y_test, y_pred):
    # Fixed: Restored indentation inside the visualization loop
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{true} P:{pred}")
    ax.axis('off')

plt.tight_layout()
plt.show()