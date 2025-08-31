import cv2
import numpy as np
from collections import defaultdict

class ShapeDetector:
    def __init__(self):
        self.shape_counts = defaultdict(int)
        
    def detect_shape(self, contour):
        """
        Detect the shape of a contour based on the number of vertices
        """
        # Calculate perimeter and approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Get the number of vertices
        vertices = len(approx)
        
        # Calculate area and bounding rectangle
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        
        shape = "Unknown"
        
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            # Check if it's a square or rectangle
            if 0.85 <= aspect_ratio <= 1.15:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif vertices > 4:
            # For circles, check circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                shape = "Circle"
        
        return shape, approx
    
    def reset_counts(self):
        """Reset shape counters"""
        self.shape_counts.clear()

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    detector = ShapeDetector()
    
    # Create trackbars for HSV adjustment (optional for better detection)
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Lower Threshold', 'Controls', 50, 255, lambda x: None)
    cv2.createTrackbar('Upper Threshold', 'Controls', 150, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', 1000, 10000, lambda x: None)
    cv2.createTrackbar('Max Area', 'Controls', 50000, 100000, lambda x: None)
    
    print("Shape Detection Started!")
    print("Controls:")
    print("- 'q' or 'ESC': Quit")
    print("- 'r': Reset counters")
    print("- Adjust trackbars for better detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        original = frame.copy()
        
        # Get trackbar values
        lower_thresh = cv2.getTrackbarPos('Lower Threshold', 'Controls')
        upper_thresh = cv2.getTrackbarPos('Upper Threshold', 'Controls')
        min_area = cv2.getTrackbarPos('Min Area', 'Controls')
        max_area = cv2.getTrackbarPos('Max Area', 'Controls')
        
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, lower_thresh, upper_thresh)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset counts for this frame
        detector.reset_counts()
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter contours by area (to focus on hand-held objects)
            if min_area < area < max_area:
                # Detect shape
                shape, approx = detector.detect_shape(contour)
                
                if shape != "Unknown":
                    # Increment counter
                    detector.shape_counts[shape] += 1
                    
                    # Get bounding rectangle for labeling
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Choose color based on shape
                    colors = {
                        "Circle": (0, 255, 0),      # Green
                        "Rectangle": (255, 0, 0),   # Blue
                        "Square": (0, 0, 255),      # Red
                        "Triangle": (255, 255, 0)   # Cyan
                    }
                    color = colors.get(shape, (255, 255, 255))
                    
                    # Draw contour and label
                    cv2.drawContours(frame, [contour], -1, color, 2)
                    cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)
                    
                    # Add shape label
                    cv2.putText(frame, shape, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Add area info
                    cv2.putText(frame, f"Area: {int(area)}", (x, y + h + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Display counters on the frame
        y_offset = 30
        cv2.putText(frame, "Shape Counts:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        shapes_order = ["Circle", "Rectangle", "Square", "Triangle"]
        colors = {
            "Circle": (0, 255, 0),
            "Rectangle": (255, 0, 0),
            "Square": (0, 0, 255),
            "Triangle": (255, 255, 0)
        }
        
        for i, shape in enumerate(shapes_order):
            count = detector.shape_counts[shape]
            y_pos = y_offset + (i + 1) * 25
            color = colors[shape]
            cv2.putText(frame, f"{shape}: {count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset counters", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (255, 255, 255), 1)
        
        # Show frames
        cv2.imshow('Shape Detection', frame)
        cv2.imshow('Edges', edges)
        cv2.imshow('Original', original)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break
        elif key == ord('r'):  # Reset counters
            detector.reset_counts()
            print("Counters reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Shape detection ended.")

if __name__ == "__main__":
    main()