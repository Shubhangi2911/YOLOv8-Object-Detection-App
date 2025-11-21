🧠 YOLOv8 & YOLO11 Object Detection Web App

This project is a Flask-based web application that performs real-time and image-based object detection using the latest YOLOv8 & YOLO11 models from Ultralytics.
It allows users to upload an image or use live webcam feed to detect objects quickly and accurately.

🎯 Overview

Object detection helps identify and locate multiple objects in an image or video.
This app demonstrates:

Image upload detection

Live webcam detection

Bounding boxes, class labels, and confidence scores

YOLOv8 & YOLO11 detection comparison

Clean, simple web interface built using Flask

🥅 Objectives

Build a simple and interactive object detection web app

Detect people, animals, vehicles, and more using YOLO

Provide real-time predictions with high accuracy

Offer both Image Detection & Live Detection (Webcam) modes

Demonstrate deep learning computer vision with YOLOv8/YOLO11

📂 Dataset

YOLOv8 & YOLO11 models are pre-trained on COCO Dataset, which includes:

80 object classes

118,000 training images

5,000 validation images

Example Object Classes:

Person

Car

Horse

Bicycle

Dog

Cat

Bus

Truck

Bottle

Laptop

🧩 Example Features Detected

Your model detects objects like:

Humans

Horses

Vehicles

Everyday items

Animals

And many more (COCO dataset classes)

🖼️ Detection Example

Below is an example of original vs detected image from this app:

🔹 Original vs Detected Output
<img src="/mnt/data/Screenshot (345).png" alt="Detection Result" width="800">
⚙️ Technologies Used
Backend

Python

Flask

Ultralytics YOLO

OpenCV

Deep Learning

YOLOv8

YOLO11

Frontend

HTML

CSS

Bootstrap

🧠 Model Architecture (YOLOv8 / YOLO11)

YOLO Model Pipeline:

Input Layer – Accepts image/frame

Backbone (CSP/Transformer) – Extracts deep visual features

Neck (FPN/PAN) – Combines features at different scales

Head – Predicts:

Bounding Boxes

Object Class

Confidence Score

YOLOv8 & YOLO11 improvements include:

Faster inference

Higher accuracy

Better small-object detection

Transformer-enabled architecture (YOLO11)

▶️ How the App Works
Image Detection

User uploads an image

App processes image with YOLOv8/YOLO11

Displays bounding boxes + confidence scores

Live Detection

Webcam opens

YOLO model runs real-time frame-by-frame detection
