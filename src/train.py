"""Train a Decision Tree model on the Iris dataset."""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 1. Load data
data = pd.read_csv("data/iris.csv")

# 2. Separate features and target
X = data.drop("species", axis=1)
y = data["species"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create model
model = DecisionTreeClassifier(random_state=42)

# 5. Train
model.fit(X_train, y_train)

# 6. Predict
predictions = model.predict(X_test)

# 7. Evaluate
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# 8. Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
