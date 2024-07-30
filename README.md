# YOLOv8 Object Detection Project

This project aims to perform object detection on images and videos using YOLOv8.

## Project Contents

### Files

- **object_det_yolov8_image.py**: Performs object detection on an image.
- **object_det_yolov8_video.py**: Performs object detection on a video or webcam feed.
- **COCO_classes.json**: Contains the COCO dataset class names.

### Video Example

Here is a video example:

https://github.com/user-attachments/assets/b7089326-d7b3-4616-9d72-0e515fecba36
https://github.com/user-attachments/assets/401ba116-8da1-4ae3-9015-897d3646617e


<video width="300" height="360" controls>
  <source src="videos\input_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
<video width="300" height="360" controls>
  <source src="videos\output_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Image Example

Here is a image example:

![input image](images\bears.jpg)
![predicted image](images\predicted_bears.jpg)

### Requirements

Before running the project, you need to install the following Python libraries:

- ultralytics
- numpy
- opencv-python
- json

Additionally, download the YOLOv8 model file (`yolov8n.pt`) and place it in the root directory of your project.

### Installation

You can install the required libraries using the following command:

```bash
pip install ultralytics numpy opencv-python 
