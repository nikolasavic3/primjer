import pickle
import numpy as np
import cv2


# Load model - THIS CAN FAIL!
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = np.array([[0.5, -0.2, 0.1, 0.8]], dtype=np.float)
print(f"Prediction: {model.predict(sample)}")

cap = cv2.VideoCapture(0)  # or a video file
print(f"OpenCV version: {cv2.__version__}")
print(f"Camera opened: {cap.isOpened()}")