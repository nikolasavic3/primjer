# Realistic ML Reproducibility Issue:
# Train on Mac, deploy on Linux server = DIFFERENT PREDICTIONS!

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

# Load real dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

if os.path.exists('trained_model.pkl'):
    # DEPLOYMENT: Load model trained on another machine
    print("Loading model trained on another machine...")
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"\nPredictions: {predictions}")
    print(f"Probabilities[0]: {probabilities[0]}")
    print(f"Probability sum: {probabilities.sum():.10f}")
    
    # Compare with expected (saved during training)
    if os.path.exists('expected_predictions.npy'):
        expected = np.load('expected_predictions.npy')
        if np.array_equal(predictions, expected):
            print("\n✅ Predictions MATCH training machine!")
        else:
            diff = (predictions != expected).sum()
            print(f"\n❌ {diff} predictions DIFFER from training machine!")
            print("This can happen due to different BLAS/floating point!")
else:
    # TRAINING: Train and save model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save expected predictions
    predictions = model.predict(X_test)
    np.save('expected_predictions.npy', predictions)
    
    print(f"Accuracy: {(predictions == y_test).mean():.2%}")
    print(f"Predictions: {predictions}")
    print(f"\n✅ Model saved! Now run on another machine to compare.")
