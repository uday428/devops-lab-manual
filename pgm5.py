import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def generate_data():
    np.random.seed(42)  # For reproducibility
    x = np.random.rand(100)  # Generate 100 random values in [0,1]
    # Label first 50 points
    labels = np.array(["Class1" if xi <= 0.5 else "Class2" for xi in x[:50]])
    return x, labels

def knn_classification(train_x, train_labels, test_x, k):
    predictions = []
    for x_test in test_x:
        distances = np.abs(train_x - x_test)  # Compute distance
        nearest_indices = np.argsort(distances)[:k]  # k nearest points
        nearest_labels = train_labels[nearest_indices]  # Labels of neighbors
        
        # Majority voting
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

def main():
    # Everything inside main() must be indented!
    x, labels = generate_data()
    train_x = x[:50]
    test_x = x[50:]
    train_labels = labels
    
    k_values = [1, 2, 3, 4, 5, 20, 30]
    results = {}
    
    for k in k_values:
        predictions = knn_classification(train_x, train_labels, test_x, k)
        results[k] = predictions
        
    # Print results
    for k, preds in results.items():
        print(f"\nResults for k = {k}")
        print(preds)
        
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(train_x, [1]*50,
                c=["blue" if lbl=="Class1" else "red" for lbl in train_labels],
                label="Training Data", edgecolors='black', s=100)
                
    for k, preds in results.items():
        plt.scatter(test_x, [k]*50,
                    c=["blue" if lbl=="Class1" else "red" for lbl in preds],
                    alpha=0.6)
                    
    plt.xlabel("x values")
    plt.ylabel("k values")
    plt.title("KNN Classification Results for Different k Values")
    plt.show()

if __name__ == "__main__":
    main()