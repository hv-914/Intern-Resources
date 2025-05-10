import cv2
import numpy as np

class LaneDetector:
    def __init__(self, lane_width_cm=15, border_width_cm=1.5):
        # Store lane specifications
        self.lane_width_cm = lane_width_cm
        self.border_width_cm = border_width_cm
        
        # Define color ranges for detection
        # Yellow range in HSV - widened range for better detection
        self.yellow_lower = np.array([10, 70, 130])
        self.yellow_upper = np.array([50, 180, 255])
        
        # Black range in HSV - adjusted for better detection
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 70])
        
        # Direction status-
        self.direction = "straight"
        self.last_reported_direction = None  # Track last reported direction for terminal output
        
        # Add detection thresholds as adjustable parameters
        self.yellow_threshold = 1  # Lower threshold for better sensitivity
        self.black_threshold = 1  # Lower threshold for better sensitivity
        
        # Add smoothing for direction changes
        self.prev_directions = ["straight"] * 5  # Store last 5 directions for smoothing

    def process_frame(self, frame):
        """Process a video frame to detect lanes and determine direction"""
        # Create a copy of the frame
        result_frame = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Define ROI (bottom half of the frame)
        roi_height = height // 2
        roi = frame[height - roi_height:height, 0:width]
        
        # Convert ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create masks for yellow and black colors
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # Define three vertical sections for direction detection (33-34-33 split)
        left_boundary = width // 3  # 33% of width
        right_boundary = (width * 2) // 3  # 67% of width (33% + 34%)
        
        # Create three ROIs for direction detection
        left_roi = yellow_mask[:, 0:left_boundary]
        center_roi = black_mask[:, left_boundary:right_boundary]
        right_roi = yellow_mask[:, right_boundary:width]
        
        # Calculate the percentage of yellow and black in each section
        left_yellow_percent = np.sum(left_roi) / (left_roi.size * 255) * 100 if left_roi.size > 0 else 0
        center_black_percent = np.sum(center_roi) / (center_roi.size * 255) * 100 if center_roi.size > 0 else 0
        right_yellow_percent = np.sum(right_roi) / (right_roi.size * 255) * 100 if right_roi.size > 0 else 0
        
        # Determine current direction based on yellow and black percentages
        current_direction = "unknown"
        
        # Decision logic for direction
        if center_black_percent > self.black_threshold:
            if left_yellow_percent > self.yellow_threshold and right_yellow_percent > self.yellow_threshold:
                current_direction = "straight"
            elif left_yellow_percent > self.yellow_threshold:
                current_direction = "right"
            elif right_yellow_percent > self.yellow_threshold:
                current_direction = "left"
        
        # Update direction history
        self.prev_directions.pop(0)
        self.prev_directions.append(current_direction)
        
        # Apply smoothing to reduce flickering
        prev_direction = self.direction
        if current_direction != "unknown":
            if self.prev_directions.count(current_direction) >= 3:  # If majority of recent directions match
                self.direction = current_direction
        
        # Print direction changes to terminal
        if self.direction != self.last_reported_direction:
            print(f"Direction: {self.direction.upper()}")
            print(f"  Left yellow: {left_yellow_percent:.1f}%, Center black: {center_black_percent:.1f}%, Right yellow: {right_yellow_percent:.1f}%")
            self.last_reported_direction = self.direction
        
        # Visualize the detection - keeping ROI and section lines
        self._draw_visualization(result_frame, roi_height, left_boundary, right_boundary)
        
        # Display detection values (L=Left yellow, C=Center black, R=Right yellow)
        cv2.putText(result_frame, f"L: {left_yellow_percent:.1f}%", (10, height - roi_height - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"C: {center_black_percent:.1f}%", (10, height - roi_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"R: {right_yellow_percent:.1f}%", (10, height - roi_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add threshold lines to help with tuning
        cv2.putText(result_frame, f"Y-Thresh: {self.yellow_threshold}%", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(result_frame, f"B-Thresh: {self.black_threshold}%", (width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return result_frame
    
    def _draw_visualization(self, frame, roi_height, left_boundary, right_boundary):
        """Draw visualization elements on the frame"""
        height, width = frame.shape[:2]
        
        # Draw ROI boundary
        cv2.line(frame, (0, height - roi_height), (width, height - roi_height), (0, 255, 0), 2)
        
        # Draw vertical section lines
        cv2.line(frame, (left_boundary, height - roi_height), (left_boundary, height), (0, 255, 0), 2)
        cv2.line(frame, (right_boundary, height - roi_height), (right_boundary, height), (0, 255, 0), 2)
        
        # Display direction
        direction_color = (0, 0, 255)  # Default red
        if self.direction == "straight":
            direction_color = (0, 255, 0)  # Green
        elif self.direction == "unknown":
            direction_color = (0, 165, 255)  # Orange
            
        cv2.putText(frame, f"Direction: {self.direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, direction_color, 2)

    def adjust_thresholds(self, yellow_adj=0, black_adj=0):
        """Allow dynamic adjustment of thresholds"""
        self.yellow_threshold = max(1, self.yellow_threshold + yellow_adj)
        self.black_threshold = max(1, self.black_threshold + black_adj)
        print(f"Thresholds updated - Yellow: {self.yellow_threshold}%, Black: {self.black_threshold}%")

def main():
    # Initialize webcam or video source
    cap = cv2.VideoCapture(0)  # 0 for default webcam, or provide a video file path
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera resolution (try different resolutions for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create lane detector with your specifications
    detector = LaneDetector(lane_width_cm=15, border_width_cm=1.5)
    
    print("\n=== Lane Detection Started ===")
    print("Detection Logic:")
    print("  STRAIGHT: Center is black, both left and right have yellow")
    print("  LEFT: Center is black, right has yellow")
    print("  RIGHT: Center is black, left has yellow")
    print("\nControls:")
    print("  'q' - Quit")
    print("  '+'/'-' - Adjust yellow threshold")
    print("  '['/']' - Adjust black threshold")
    print("\nCurrent thresholds - Yellow: 5%, Black: 10%")
    print("\n=== Direction Changes ===")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frame
        result = detector.process_frame(frame)
        
        # Display result
        cv2.imshow("Lane Detection", result)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            detector.adjust_thresholds(yellow_adj=1)
        elif key == ord('-'):
            detector.adjust_thresholds(yellow_adj=-1)
        elif key == ord(']'):
            detector.adjust_thresholds(black_adj=1)
        elif key == ord('['):
            detector.adjust_thresholds(black_adj=-1)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Lane Detection Stopped ===")

if __name__ == "__main__":
    main()