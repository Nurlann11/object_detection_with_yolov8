# libraries:
from ultralytics import YOLO
import numpy as np
import cv2
import random
import json

# Load COCO class names
with open('COCO_classes.json', 'r') as f:
    class_names = json.load(f)

# Generate random colors for each class 
colors = []
for i in range(80): 
    b = random.randint(100, 255)
    g = random.randint(100, 255)
    r = random.randint(100, 255)
    colors.append((b, g, r))
 
# Load the YOLO model
model = YOLO('weights/yolov8n.pt', 'v8')

# Define the video source
cap = cv2.VideoCapture('./videos/input_video.mp4')
if not cap.isOpened():
    print('Cannot open camera')
    exit()

# Get the original video's frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('./videos/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end)")
        break


    # Perform object detection
    detect_params = model.predict(source=frame, conf=0.25, save=False)

    if len(detect_params) != 0:
        results = detect_params[0]
        boxes = results.boxes.xyxy  # Get the bounding box coordinates
        confidences = results.boxes.conf  # Get the confidence scores
        class_ids = results.boxes.cls  # Get the class IDs

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy().astype(int)  # Convert boxes to numpy array
            conf = confidences[i].cpu().numpy()  # Get the confidence score
            class_id = int(class_ids[i].cpu().numpy())  # Get the class ID
            label = f"{class_names[str(class_id)]} {conf:.2f}"  # Create the label

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colors[class_id], 1)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

    # Display the frame
    cv2.imshow('Object_Detection', frame)
    
    # Write the frame into the output video file
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and writer objects and close display windows
cap.release()
out.release()
cv2.destroyAllWindows()
