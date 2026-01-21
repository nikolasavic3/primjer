import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# Create dummy data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained with sklearn {__import__('sklearn').__version__}")
print(f"NumPy version: {np.__version__}")

