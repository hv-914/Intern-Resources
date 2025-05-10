import cv2
import numpy as np
import cv2
from ultralytics import YOLO

# Global variables
left_lane_history = []
right_lane_history = []
steering_history = []
model = YOLO("C:/Users/merit/PROJECTS/aiLite_turnSigns.pt") 

class LaneDetectionUI:
    def __init__(self):
        self.recording = False
        self.threshold_values = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 50,
            'min_line_length': 40,
            'max_line_gap': 150
        }

def draw_roi(frame, vertices):
    """Draw ROI as a semi-transparent blue rectangle"""
    overlay = frame.copy()
    cv2.fillPoly(overlay, vertices, (255, 100, 0))  # Blue color (BGR)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # 20% opacity
    cv2.polylines(frame, vertices, True, (255, 100, 0), 2)  # Draw border
    return frame

def preprocess_frame(frame, ui):
    """Basic frame preprocessing"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, ui.threshold_values['canny_low'], 
                     ui.threshold_values['canny_high'])
    return edges

def apply_roi_mask(edges, vertices):
    """Apply a mask to focus on the region of interest"""
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def detect_vertical_lines(edges, ui):
    """Detect vertical lines using Hough transform"""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                           threshold=ui.threshold_values['hough_threshold'],
                           minLineLength=ui.threshold_values['min_line_length'],
                           maxLineGap=ui.threshold_values['max_line_gap'])
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                if 1.5 < abs(slope) < 10:
                    vertical_lines.append(line[0])
    return vertical_lines

def average_and_extrapolate_lines(frame, lines):
    """Average and extrapolate the detected lines"""
    global left_lane_history, right_lane_history

    if not lines:
        return None, None

    height, width = frame.shape[:2]
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < 0:
            left_lines.append((slope, intercept))
        elif slope > 0:
            right_lines.append((slope, intercept))

    def get_average_line(lines):
        if not lines:
            return None
        avg_slope = np.mean([line[0] for line in lines])
        avg_intercept = np.mean([line[1] for line in lines])
        return avg_slope, avg_intercept

    left_lane = get_average_line(left_lines)
    right_lane = get_average_line(right_lines)

    def smooth_lane(lane_history, new_lane, max_history=5):
        if new_lane is not None:
            lane_history.append(new_lane)
            if len(lane_history) > max_history:
                lane_history.pop(0)
        if lane_history:
            return np.mean([lane[0] for lane in lane_history]), np.mean([lane[1] for lane in lane_history])
        return None

    left_lane = smooth_lane(left_lane_history, left_lane)
    right_lane = smooth_lane(right_lane_history, right_lane)

    def create_line_points(slope, intercept):
        if slope is None or intercept is None:
            return None
        y1 = height
        y2 = int(height * 0.6)
        if slope == 0:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    left_line = create_line_points(*left_lane) if left_lane else None
    right_line = create_line_points(*right_lane) if right_lane else None

    return left_line, right_line

def draw_lanes(frame, lanes):
    """
    Draw lane lines by fitting curves to detected points.
    
    Args:
        frame: Input image/frame
        lanes: List containing [x1, y1, x2, y2] for each lane
    Returns:
        frame: Frame with lanes drawn
    """
    for lane in lanes:
        if lane is not None:
            x1, y1, x2, y2 = lane
            
            # Create more points between the endpoints
            num_points = 10
            x_points = np.linspace(x1, x2, num_points)
            y_points = np.linspace(y1, y2, num_points)
            
            # Add some intermediate points with slight variation to detect curves
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            x_points = np.append(x_points, x_mid)
            y_points = np.append(y_points, y_mid)
            
            # Fit polynomials
            coeffs_linear = np.polyfit(y_points, x_points, 1)
            coeffs_quadratic = np.polyfit(y_points, x_points, 2)
            
            # Calculate R-squared for both fits
            y_linear = np.polyval(coeffs_linear, y_points)
            y_quadratic = np.polyval(coeffs_quadratic, y_points)
            
            r2_linear = 1 - np.sum((x_points - y_linear) ** 2) / np.sum((x_points - np.mean(x_points)) ** 2)
            r2_quadratic = 1 - np.sum((x_points - y_quadratic) ** 2) / np.sum((x_points - np.mean(x_points)) ** 2)
            
            # Choose between linear and quadratic fit
            if r2_quadratic > r2_linear + 0.1:
                coeffs = coeffs_quadratic
                degree = 2
            else:
                coeffs = coeffs_linear
                degree = 1
            
            # Generate points for drawing
            y_values = np.linspace(min(y_points), max(y_points), num=50)
            x_values = np.polyval(coeffs, y_values)
            
            # Draw the lane
            points = np.column_stack((x_values.astype(np.int32), y_values.astype(np.int32)))
            for i in range(len(points) - 1):
                cv2.line(frame, 
                        (points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1]), 
                        (0, 255, 255), 
                        5)
                
    return frame

def calculate_steering(frame, lanes):
    """Calculate steering position and advice with deviation"""
    height, width = frame.shape[:2]
    frame_center = width // 2

    if not lanes or (lanes[0] is None and lanes[1] is None):
        return frame_center, "No Lane Detected", 0

    left_lane = lanes[0]
    right_lane = lanes[1]

    if left_lane and right_lane:
        left_x = (left_lane[0] + left_lane[2]) // 2
        right_x = (right_lane[0] + right_lane[2]) // 2
        lane_center = (left_x + right_x) // 2
    elif left_lane:
        lane_center = (left_lane[0] + left_lane[2]) // 2
    elif right_lane:
        lane_center = (right_lane[0] + right_lane[2]) // 2
    else:
        return frame_center, "No Lane Detected", 0

    deviation = lane_center - frame_center
    
    if deviation < -25:
        temp = abs(deviation + 25)
        temp = (temp//10) + 1
        advice = f"Turn Left ({temp})"
        print(f'l{temp}')
    elif deviation > 25:
        temp = deviation - 25
        temp = (temp//10) + 1
        advice = f"Turn Right ({temp})"
        print(f'r{temp}')
    else:
        advice = "Go Straight"
        print(f'0')

    return lane_center, advice, deviation

def draw_steering_guidance(frame, lane_center, advice, deviation):
    """Draw steering guidance with I-beam visualization and deviation"""
    height, width = frame.shape[:2]
    beam_y = int(height * 0.85)
    beam_width = 200
    
    # Calculate beam boundaries
    left_boundary = width//2 - beam_width//2
    right_boundary = width//2 + beam_width//2
    
    # Draw main guidance beam
    cv2.line(frame, (left_boundary, beam_y), 
             (right_boundary, beam_y), (0, 255, 120), 3)
    
    # Draw partition lines
    left_x = width//2 - beam_width//8
    right_x = width//2 + beam_width//8
    
    partition_height = 12
    cv2.line(frame, (left_x, beam_y - partition_height),
             (left_x, beam_y + partition_height), (0, 255, 120), 3)
    cv2.line(frame, (right_x, beam_y - partition_height),
             (right_x, beam_y + partition_height), (0, 255, 120), 3)
    
    # Draw endpoint vertical lines
    cv2.line(frame, (left_boundary, beam_y - partition_height),
             (left_boundary, beam_y + partition_height), (0, 255, 120), 3)
    cv2.line(frame, (right_boundary, beam_y - partition_height),
             (right_boundary, beam_y + partition_height), (0, 255, 120), 3)
    
    # Draw steering indicator with smoothing and boundary constraints
    steering_history.append(lane_center)
    if len(steering_history) > 5:
        steering_history.pop(0)
    smooth_center = sum(steering_history) / len(steering_history)
    
    # Constrain indicator position within beam boundaries
    constrained_center = max(left_boundary, min(right_boundary, int(smooth_center)))
    
    # Draw indicator circles
    cv2.circle(frame, (constrained_center, beam_y), 6, (0, 0, 255), -1)
    cv2.circle(frame, (constrained_center, beam_y), 10, (0, 0, 255), 2)
    
    # Draw advice text above the I-beam
    text_size = cv2.getTextSize(advice, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = width // 2 - (text_size[0] // 2)
    text_y = beam_y - 40
    cv2.putText(frame, advice, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw deviation text in the top-left corner
    deviation_text = f"Deviation: {abs(deviation)}px"
    if deviation != 0:
        deviation_text += " Left" if deviation < 0 else " Right"
    cv2.putText(frame, deviation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def main():
    ui = LaneDetectionUI()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy for YOLO detections
        yolo_frame = frame.copy()

        # Run YOLOv8 inference (suppress logs)
        results = model(frame, verbose=False)  # Disable detailed logging

        detected_labels = set()  # Store unique labels

        # Draw detections on the yolo_frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cls = int(box.cls[0].item())  # Class index
                label = model.names[cls]

                # Store unique label for terminal output
                detected_labels.add(label)

                # Draw rectangle and label on yolo_frame
                cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(yolo_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print only the labels in the terminal
        #if detected_labels:
        #    print(", ".join(detected_labels))  # Prints only unique labels
        
        # Show YOLO Detections in a separate window
        cv2.imshow('YOLO Detections', cv2.resize(yolo_frame, (720, 540)))

        if 'stop' in detected_labels:
            print('stop')
        elif 'right' in detected_labels:
            print('right')
        elif 'left' in detected_labels:
            print('left')
        else:
            # Process frame for lane detection
            height, width = frame.shape[:2]

            # Rectangle
            roi_vertices = np.array([[
                (0, height // 2), # Top-left corner of the rectangle
                (width, height // 2),  # Top-right corner
                (width, height),  # Bottom-right corner
                (0, height)  # Bottom-left corner
            ]], dtype=np.int32)

            # Draw ROI first
            frame = draw_roi(frame.copy(), roi_vertices)
            
            edges = preprocess_frame(frame, ui)
            roi_edges = apply_roi_mask(edges, roi_vertices)
            vertical_lines = detect_vertical_lines(roi_edges, ui)
            extrapolated_lines = average_and_extrapolate_lines(frame, vertical_lines)
            
            # Draw lanes and steering guidance
            frame_with_lanes = draw_lanes(frame.copy(), extrapolated_lines)
            lane_center, advice, deviation = calculate_steering(frame, extrapolated_lines)
            final_frame = draw_steering_guidance(frame_with_lanes, lane_center, advice, deviation)
            
            # Show Lane Detection in another window
            cv2.imshow('Lane Detection System', cv2.resize(final_frame, (720, 540)))
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            ui.recording = not ui.recording
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    