import cv2
import numpy as np

def calculate_steering_angle(frame, line_contour):
    height, width = frame.shape[:2]
    M = cv2.moments(line_contour)
    
    # Calculate centroid of the line
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
    else:
        cx = width // 2
    
    # Calculate offset from center
    center_offset = cx - (width // 2)
    
    # Convert offset to steering angle - adjusted for smaller line
    # Increased sensitivity for finer control with thin line
    steering_angle = -float(center_offset) * 0.2  # Increased sensitivity
    
    return steering_angle, cx

def apply_steering_control(steering_angle):
    # Tighter threshold for more precise control
    if abs(steering_angle) < 5:  # Reduced threshold
        return "Go Straight"
    elif abs(steering_angle) < 15:
        return "Slight " + ("Left" if steering_angle > 0 else "Right")
    else:
        return "Sharp " + ("Left" if steering_angle > 0 else "Right")

cap = cv2.VideoCapture(0)

# Adjusted threshold for 1-inch line detection
# For a 1-inch line, we need a much smaller area threshold
MIN_CONTOUR_AREA = 5000  # Reduced from 75000
MAX_CONTOUR_AREA = 50000  # Added upper limit to filter out larger objects

# For line filtering
MIN_LINE_WIDTH = 15   # Minimum width in pixels for our 1-inch line
MAX_LINE_WIDTH = 50   # Maximum width in pixels for our 1-inch line

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Apply binary threshold - adjusted for better line detection
    # You may need to adjust threshold value based on lighting conditions
    _, binary_frame = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)
    
    # Define region of interest (ROI) - focusing on bottom half of frame
    roi_height = height // 3  # Lower third of the frame
    roi_vertices = np.array([
        [(0, height - roi_height), (width, height - roi_height), 
         (width, height), (0, height)]
    ], dtype=np.int32)
    
    # Create mask and apply ROI
    mask = np.zeros_like(binary_frame)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_result = cv2.bitwise_and(binary_frame, mask)
    
    # Find contours in the ROI
    contours, _ = cv2.findContours(roi_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create overlay for visualization
    overlay = np.zeros_like(frame)
    highlighted_frame = frame.copy()
    
    # Filter contours by area to find our 1-inch line
    valid_contours = [
        cnt for cnt in contours 
        if MIN_CONTOUR_AREA < cv2.contourArea(cnt) < MAX_CONTOUR_AREA
    ]
    
    # Further filter by aspect ratio to identify line-like shapes
    line_contours = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # If contour width is within our expected range for a 1-inch line
        if MIN_LINE_WIDTH < w < MAX_LINE_WIDTH or MIN_LINE_WIDTH < h < MAX_LINE_WIDTH:
            line_contours.append(cnt)
    
    if line_contours:
        # Get the largest line contour (should be our 1-inch line)
        line_contour = max(line_contours, key=cv2.contourArea)
        
        # Highlight the detected line
        cv2.fillPoly(overlay, [line_contour], (0, 255, 0))
        highlighted_frame = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
        cv2.drawContours(highlighted_frame, [line_contour], -1, (0, 255, 255), 2)
        
        # Calculate steering angle
        steering_angle, center_x = calculate_steering_angle(frame, line_contour)
        steering_direction = apply_steering_control(steering_angle)
        
        # Draw guiding point and line
        cv2.circle(highlighted_frame, (center_x, height - 30), 5, (0, 0, 255), -1)
        cv2.line(highlighted_frame, (width // 2, height - 30), (center_x, height - 30), (0, 0, 255), 2)
        
        # Display steering direction
        text_size = cv2.getTextSize(steering_direction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = height - 60
        cv2.putText(highlighted_frame, steering_direction, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display angle value
        angle_text = f"Angle: {steering_angle:.1f}"
        cv2.putText(highlighted_frame, angle_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        # No line detected
        message = "No Line Detected"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = width // 2 - text_size[0] // 2
        text_y = height // 2
        cv2.putText(highlighted_frame, message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw ROI outline
    cv2.polylines(highlighted_frame, [roi_vertices], True, (255, 0, 0), 2)
    
    # Display the frames
    cv2.imshow('Binary View', cv2.resize(binary_frame, (640, 480)))
    cv2.imshow('Line Following', cv2.resize(highlighted_frame, (640, 480)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()