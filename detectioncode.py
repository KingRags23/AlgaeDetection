import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Load the image dataset
dataset_path = "/path/to/dataset/folder"
algae_images = []
nonalgae_images = []
for file in os.listdir(dataset_path):
    if file.startswith("algae"):
        img = cv2.imread(os.path.join(dataset_path, file))
        algae_images.append(img)
    else:
        img = cv2.imread(os.path.join(dataset_path, file))
        nonalgae_images.append(img)

# Prepare the dataset for machine learning
X = []
y = []
for img in algae_images:
    X.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten())
    y.append(1)
for img in nonalgae_images:
    X.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten())
    y.append(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear SVM on the dataset
svm = LinearSVC()
svm.fit(X_train, y_train)

# Capture an image from the drone
drone_camera = cv2.VideoCapture(0)
ret, frame = drone_camera.read()

# Preprocess the image for prediction
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_flat = gray.flatten()

# Make a prediction using the trained SVM
prediction = svm.predict([gray_flat])

# Print the prediction result
if prediction == 1:
    print("Algae detected in the water!")
else:
    print("No algae detected in the water.")
