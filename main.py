# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 2. Load dataset
iris = load_iris()

# Convert dataset into DataFrame (for better understanding)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column (flower type)
df['target'] = iris.target

# 3. Show first 5 rows of data
print(df.head())

# 4. Split features (X) and labels (y)
X = df.drop('target', axis=1)   # input features
y = df['target']               # output labels

# 5. Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature scaling (important for ML models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Create model
model = LogisticRegression()

# 8. Train model
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 12. Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 13. Simple prediction example
sample = [[5.1, 3.5, 1.4, 0.2]]  # example flower measurements
sample = scaler.transform(sample)

prediction = model.predict(sample)
print("Predicted class:", iris.target_names[prediction][0])

