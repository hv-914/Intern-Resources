import cv2
import numpy as np
import time

class LaneDetector:
    def __init__(self):
        # Define color range for black detection in HSV
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 80])
        
        # Direction status
        self.direction = "straight"
        self.last_reported_direction = None
        
        # Detection threshold
        self.black_threshold = 1.5
        
        # Enhanced smoothing for direction changes
        self.prev_directions = ["straight"] * 10
        
        # Confidence tracking for more stable direction changes
        self.direction_confidence = {
            "straight": 0.0,
            "left": 0.0,
            "right": 0.0,
            "unknown": 0.0
        }
        self.confidence_decay = 0.7
        self.confidence_threshold = 3.0
        
        # Adaptive threshold variables
        self.is_adaptive_mode = True
        self.frames_processed = 0
        self.black_history = []
        self.history_max_size = 30
        
        # Frame preprocessing parameters
        self.blur_kernel_size = 5
        self.use_contrast_enhancement = True
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process_frame(self, frame):
        """Process a video frame to detect lanes and determine direction"""
        # Create a copy of the frame
        result_frame = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Define ROI (bottom half of the frame)
        roi_height = height // 2
        roi = frame[height - roi_height:height, 0:width]
        
        # Apply preprocessing to improve detection quality
        roi_processed = self._preprocess_image(roi)
        
        # Convert processed ROI to HSV color space
        hsv = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2HSV)
        
        # Create mask for black color
        black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        
        # Define three regions for direction detection
        left_boundary = width // 3
        right_boundary = (width * 2) // 3
        
        # Create three ROIs for direction detection
        left_roi = black_mask[:, 0:left_boundary]
        center_roi = black_mask[:, left_boundary:right_boundary]
        right_roi = black_mask[:, right_boundary:width]
        
        # Calculate the percentage of black in each section
        left_black_percent = np.sum(left_roi) / (left_roi.size * 255) * 100 if left_roi.size > 0 else 0
        center_black_percent = np.sum(center_roi) / (center_roi.size * 255) * 100 if center_roi.size > 0 else 0
        right_black_percent = np.sum(right_roi) / (right_roi.size * 255) * 100 if right_roi.size > 0 else 0
        
        # Update color percentage history for adaptive thresholding
        if self.is_adaptive_mode:
            self._update_color_history(left_black_percent, right_black_percent)
            
            # Periodically adjust thresholds based on history
            if self.frames_processed % 10 == 0 and self.frames_processed > 0:
                self._adjust_adaptive_thresholds()
        
        # Determine current direction with confidence based on black percentages
        current_direction, confidence = self._determine_direction(
            left_black_percent, center_black_percent, right_black_percent)
        
        # Update direction history
        self.prev_directions.pop(0)
        self.prev_directions.append(current_direction)
        
        # Update confidence values with decay
        for direction in self.direction_confidence:
            if direction == current_direction:
                self.direction_confidence[direction] += confidence
            else:
                self.direction_confidence[direction] *= self.confidence_decay
        
        # Apply improved smoothing based on confidence values
        max_confidence_direction = max(
            self.direction_confidence.items(), 
            key=lambda x: x[1]
        )
        
        prev_direction = self.direction
        
        # Only change direction if confidence is above threshold and direction has support in history
        if (max_confidence_direction[1] >= self.confidence_threshold and 
            self.prev_directions.count(max_confidence_direction[0]) >= 5):
            self.direction = max_confidence_direction[0]
            
            # Reset confidences of other directions when direction changes
            if prev_direction != self.direction:
                for dir_key in self.direction_confidence:
                    if dir_key != self.direction:
                        self.direction_confidence[dir_key] = 0.0
        
        # Print direction changes to terminal
        if self.direction != self.last_reported_direction:
            print(f"Direction: {self.direction.upper()}")
            print(f"  Left black: {left_black_percent:.1f}%, Center black: {center_black_percent:.1f}%, Right black: {right_black_percent:.1f}%")
            print(f"  Confidence: {self.direction_confidence[self.direction]:.1f}")
            self.last_reported_direction = self.direction
        
        # Visualize the detection
        self._draw_visualization(result_frame, roi_height, left_boundary, right_boundary, 
                                left_black_percent, center_black_percent, right_black_percent)
        
        # Increment frame counter
        self.frames_processed += 1
        
        return result_frame
    
    def _preprocess_image(self, image):
        """Apply preprocessing to improve image quality for detection"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Apply contrast enhancement if enabled
        if self.use_contrast_enhancement:
            # Convert to LAB color space
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            
            # Split the LAB image into L, A, and B channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l = self.clahe.apply(l)
            
            # Merge the CLAHE enhanced L channel with the original A and B channels
            lab = cv2.merge((l, a, b))
            
            # Convert back to BGR color space
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        
        return blurred
    
    def _determine_direction(self, left_black, center_black, right_black):
        """Determine direction with confidence based on black percentages"""
        # Start with unknown direction and zero confidence
        direction = "unknown"
        confidence = 0.0
        
        # Decision logic based on black regions
        # If there's significant black in both left and right with similar amounts, go straight
        # If there's more black on the left, turn right
        # If there's more black on the right, turn left
        
        # Check if we have enough black regions detected first
        if left_black > self.black_threshold or right_black > self.black_threshold:
            # Calculate the difference between left and right black percentage
            difference = abs(left_black - right_black)
            
            # If difference is small, go straight
            if difference < 3.0 and left_black > self.black_threshold and right_black > self.black_threshold:
                direction = "straight"
                confidence = min(1.0, (left_black + right_black) / 20)
            # If more black on left, turn right
            elif left_black > right_black * 1.5 and left_black > self.black_threshold:
                direction = "right"
                confidence = min(1.0, left_black / 10)
            # If more black on right, turn left
            elif right_black > left_black * 1.5 and right_black > self.black_threshold:
                direction = "left"
                confidence = min(1.0, right_black / 10)
        
        # Scale confidence to make it more meaningful
        confidence = confidence * 3.0
        
        return direction, confidence
    
    def _update_color_history(self, left_black, right_black):
        """Update color percentage history for adaptive thresholding"""
        # Add current values to history
        self.black_history.append((left_black + right_black) / 2)
        
        # Keep history at maximum size
        if len(self.black_history) > self.history_max_size:
            self.black_history.pop(0)
    
    def _adjust_adaptive_thresholds(self):
        """Dynamically adjust thresholds based on historical color data"""
        if len(self.black_history) < 10:
            return  # Not enough data yet
        
        # Calculate mean and standard deviation
        black_mean = np.mean(self.black_history)
        black_std = np.std(self.black_history)
        
        # Adjust threshold based on statistics
        self.black_threshold = max(1.0, black_mean * 0.4)
        
        print(f"Adaptive threshold adjusted - Black: {self.black_threshold:.1f}%")
    
    def _draw_visualization(self, frame, roi_height, left_boundary, right_boundary, 
                          left_black_percent, center_black_percent, right_black_percent):
        """Draw visualization elements on the frame"""
        height, width = frame.shape[:2]
        
        # Draw ROI boundary
        cv2.line(frame, (0, height - roi_height), (width, height - roi_height), (0, 255, 0), 2)
        
        # Draw vertical section lines
        cv2.line(frame, (left_boundary, height - roi_height), (left_boundary, height), (0, 255, 0), 2)
        cv2.line(frame, (right_boundary, height - roi_height), (right_boundary, height), (0, 255, 0), 2)
        
        # Display direction with color based on confidence
        confidence = self.direction_confidence[self.direction]
        direction_color = (0, 0, 255)  # Default red
        
        if self.direction == "straight" and confidence >= self.confidence_threshold:
            direction_color = (0, 255, 0)  # Green
        elif self.direction != "unknown" and confidence >= self.confidence_threshold:
            direction_color = (0, 165, 255)  # Orange
        elif self.direction == "unknown" or confidence < self.confidence_threshold:
            direction_color = (128, 128, 128)  # Gray for low confidence
            
        cv2.putText(frame, f"Direction: {self.direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, direction_color, 2)
        
        # Display confidence value
        cv2.putText(frame, f"Confidence: {confidence:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, direction_color, 2)
        
        # Display detection values (L=Left black, C=Center black, R=Right black)
        cv2.putText(frame, f"L: {left_black_percent:.1f}%", (10, height - roi_height - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"C: {center_black_percent:.1f}%", (10, height - roi_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"R: {right_black_percent:.1f}%", (10, height - roi_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add threshold information
        cv2.putText(frame, f"B-Thresh: {self.black_threshold:.1f}%", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Show adaptive mode status
        mode_text = "Adaptive" if self.is_adaptive_mode else "Manual"
        cv2.putText(frame, f"Mode: {mode_text}", (width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def adjust_thresholds(self, black_adj=0):
        """Allow manual adjustment of thresholds"""
        self.black_threshold = max(1, self.black_threshold + black_adj)
        print(f"Threshold updated - Black: {self.black_threshold:.1f}%")
    
    def toggle_adaptive_mode(self):
        """Toggle between adaptive and manual threshold modes"""
        self.is_adaptive_mode = not self.is_adaptive_mode
        mode_text = "Adaptive" if self.is_adaptive_mode else "Manual"
        print(f"Switched to {mode_text} threshold mode")
    
    def adjust_preprocessing(self, blur_adj=0, toggle_contrast=False):
        """Adjust preprocessing parameters"""
        if blur_adj != 0:
            # Ensure odd kernel size (3, 5, 7, etc.)
            self.blur_kernel_size = max(3, self.blur_kernel_size + blur_adj * 2)
            self.blur_kernel_size = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size - 1
            print(f"Blur kernel size updated: {self.blur_kernel_size}")
            
        if toggle_contrast:
            self.use_contrast_enhancement = not self.use_contrast_enhancement
            status = "enabled" if self.use_contrast_enhancement else "disabled"
            print(f"Contrast enhancement {status}")


def main():
    """Main function to run the lane detection system"""
    print("Black Lane Detection System Starting...")
    print("Controls:")
    print("  Q - Quit")
    print("  A - Toggle adaptive/manual threshold mode")
    print("  B/N - Decrease/Increase black threshold")
    print("  C - Toggle contrast enhancement")
    print("  Z/X - Decrease/Increase blur kernel size")
    print("  S - Save current frame")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify video file path
    
    # Set camera resolution if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera or video file.")
        return
    
    # Initialize lane detector
    detector = LaneDetector()
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Main loop
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process frame for lane detection
        result_frame = detector.process_frame(frame)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Black Lane Detection', result_frame)
        
        # Process key inputs
        key = cv2.waitKey(1) & 0xFF
        
        # Handle key presses
        if key == ord('q'):  # Quit
            break
        elif key == ord('a'):  # Toggle adaptive mode
            detector.toggle_adaptive_mode()
        elif key == ord('b'):  # Decrease black threshold
            detector.adjust_thresholds(black_adj=-0.5)
        elif key == ord('n'):  # Increase black threshold
            detector.adjust_thresholds(black_adj=0.5)
        elif key == ord('c'):  # Toggle contrast enhancement
            detector.adjust_preprocessing(toggle_contrast=True)
        elif key == ord('z'):  # Decrease blur kernel size
            detector.adjust_preprocessing(blur_adj=-1)
        elif key == ord('x'):  # Increase blur kernel size
            detector.adjust_preprocessing(blur_adj=1)
        elif key == ord('s'):  # Save current frame
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"black_lane_detection_{timestamp}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"Saved frame as {filename}")
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Black Lane Detection System Terminated.")


# Entry point for the script
if __name__ == "__main__":
    main()