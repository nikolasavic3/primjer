import pickle
import numpy as np

# Load model - THIS CAN FAIL!
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = np.array([[0.5, -0.2, 0.1, 0.8]], dtype=np.float)
print(f"Prediction: {model.predict(sample)}")