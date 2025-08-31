Shape Detection with OpenCV

📌 Overview

This project uses OpenCV and Python to detect geometric shapes (Circle, Rectangle, Square, Triangle) in real-time using a webcam.
It can be applied to inventory management systems, where product packaging can be classified and organized based on its shape (e.g., circular bottles, rectangular boxes, etc.).

🎯 Features

Detects Circle, Rectangle, Square, and Triangle in real-time.
Displays shape labels, contour outlines, and area information.
Shape counts are shown live on screen.
Adjustable Canny edge thresholds and area filtering using trackbars.
Reset shape counters anytime using the r key.

⚙️ Technologies Used

Python 3.x
OpenCV (cv2)
NumPy

🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/shape-detection.git

cd shape-detection


Install dependencies:

pip install opencv-python numpy


Run the script:

python shape_detection.py

Controls:

q or ESC → Quit

r → Reset counters

Trackbars (Lower Threshold, Upper Threshold, Min Area, Max Area) → Adjust for better detection

📸 Demo

Detected Shapes: Shapes are highlighted with contours and labeled.
Live Counters: Keeps track of how many of each shape are detected.
Applications: Can be integrated into inventory management systems to classify packaging (bottles, boxes, triangular packages, etc.).

🛠 Future Improvements

Train with ML/DL models for complex product detection.
Integrate with barcode/RFID systems.
Export detection results into a database for real-time inventory tracking.
