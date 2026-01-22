import numpy as np
from sklearn.linear_model import LogisticRegression
import cv2

X = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [0.2, 0.1, 0.4, 0.3],
])
y = np.array([0, 1, 1, 0])

model = LogisticRegression()
model.fit(X, y)

sample = np.array([[0.5, -0.2, 0.1, 0.8]], dtype=np.float)
print(f"Prediction: {model.predict(sample)}")

cap = cv2.VideoCapture(0)
print(f"OpenCV version: {cv2.__version__}")
print(f"Camera opened: {cap.isOpened()}")
