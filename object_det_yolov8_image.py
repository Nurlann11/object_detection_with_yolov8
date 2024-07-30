from ultralytics import YOLO
import numpy as np
import cv2

# Load YOLO v8 model
model = YOLO('yolov8n.pt', 'v8')

# Perform image prediction
image_path = './images/bears.jpg'


output = model.predict(image_path, conf=0.25, save=False)  # save=False because we will save manually

# Get the predicted image with annotations
predicted_image = output[0].plot()  # Draw predictions on the image

# Save the image to a file
output_path = './images/predicted_bears.jpg'  # Path and filename for saving
cv2.imwrite(output_path, predicted_image)

print('Predicted image saved to:', output_path)
