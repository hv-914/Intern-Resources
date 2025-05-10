import cv2
import numpy as np
from ultralytics import YOLO
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timedelta

# Global variables
bot_num = input("Enter bot number : ")
host = "192.168."+bot_num+".10" 
port = 80
left_lane_history = []
right_lane_history = []
steering_history = []
command_queue = deque(maxlen=3)
last_command_time = datetime.now()
executor = ThreadPoolExecutor(max_workers=1)
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

def send_http_request(command):
    try:
        url = f"http://{host}:{port}/?cmd={command}"
        response = requests.get(url, timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def queue_command(command):
    global last_command_time
    current_time = datetime.now()
    
    if current_time - last_command_time < timedelta(seconds=0.5):
        return
    
    if command_queue and command_queue[-1] == command:
        return
        
    command_queue.append(command)
    last_command_time = current_time
    executor.submit(send_http_request, command)

def draw_roi(frame, vertices):
    overlay = frame.copy()
    cv2.fillPoly(overlay, vertices, (255, 100, 0))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, vertices, True, (255, 100, 0), 2)
    return frame

def preprocess_frame(frame, ui):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, ui.threshold_values['canny_low'], 
                     ui.threshold_values['canny_high'])
    return edges

def apply_roi_mask(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def detect_vertical_lines(edges, ui):
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
    for lane in lanes:
        if lane is not None:
            x1, y1, x2, y2 = lane
            
            num_points = 10
            x_points = np.linspace(x1, x2, num_points)
            y_points = np.linspace(y1, y2, num_points)
            
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            x_points = np.append(x_points, x_mid)
            y_points = np.append(y_points, y_mid)
            
            coeffs_linear = np.polyfit(y_points, x_points, 1)
            coeffs_quadratic = np.polyfit(y_points, x_points, 2)
            
            y_linear = np.polyval(coeffs_linear, y_points)
            y_quadratic = np.polyval(coeffs_quadratic, y_points)
            
            r2_linear = 1 - np.sum((x_points - y_linear) ** 2) / np.sum((x_points - np.mean(x_points)) ** 2)
            r2_quadratic = 1 - np.sum((x_points - y_quadratic) ** 2) / np.sum((x_points - np.mean(x_points)) ** 2)
            
            coeffs = coeffs_quadratic if r2_quadratic > r2_linear + 0.1 else coeffs_linear
            degree = 2 if r2_quadratic > r2_linear + 0.1 else 1
            
            y_values = np.linspace(min(y_points), max(y_points), num=50)
            x_values = np.polyval(coeffs, y_values)
            
            points = np.column_stack((x_values.astype(np.int32), y_values.astype(np.int32)))
            for i in range(len(points) - 1):
                cv2.line(frame, 
                        (points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1]), 
                        (0, 255, 255), 
                        5)
    return frame

def process_lane_deviation(deviation):
    if deviation < -25:
        temp = min((abs(deviation + 25) // 10) + 1, 5)
        queue_command('l')
        print(f'l{temp}')
    elif deviation > 25:
        temp = min((deviation - 25) // 10 + 1, 5)
        queue_command('r')
        print(f'r{temp}')
    else:
        queue_command('f')
        print(f'0')

def calculate_steering(frame, lanes):
    height, width = frame.shape[:2]
    frame_center = width // 2

    if not lanes or (lanes[0] is None and lanes[1] is None):
        queue_command('s')
        print("No lanes detected - Stopping")
        return frame_center, "No Lane Detected", 0

    left_lane, right_lane = lanes[0], lanes[1]
    
    if left_lane and right_lane:
        lane_center = ((left_lane[0] + left_lane[2] + right_lane[0] + right_lane[2]) // 4)
    elif left_lane:
        lane_center = (left_lane[0] + left_lane[2]) // 2
    elif right_lane:
        lane_center = (right_lane[0] + right_lane[2]) // 2
    else:
        queue_command('s')
        print("No lanes detected - Stopping")
        return frame_center, "No Lane Detected", 0

    deviation = lane_center - frame_center
    
    if deviation < -25:
        temp = min((abs(deviation + 25) // 10) + 1, 5)
        process_lane_deviation(deviation)
        return lane_center, "Turn Left", deviation
    elif deviation > 25:
        temp = min((deviation - 25) // 10 + 1, 5)
        process_lane_deviation(deviation)
        return lane_center, "Turn Right", deviation
    else:
        process_lane_deviation(deviation)
        return lane_center, "Straight", deviation

def draw_steering_guidance(frame, lane_center, advice, deviation):
    height, width = frame.shape[:2]
    beam_y = int(height * 0.85)
    beam_width = 200
    
    left_boundary = width//2 - beam_width//2
    right_boundary = width//2 + beam_width//2
    
    cv2.line(frame, (left_boundary, beam_y), 
             (right_boundary, beam_y), (0, 255, 120), 3)
    
    left_x = width//2 - beam_width//8
    right_x = width//2 + beam_width//8
    
    partition_height = 12
    cv2.line(frame, (left_x, beam_y - partition_height),
             (left_x, beam_y + partition_height), (0, 255, 120), 3)
    cv2.line(frame, (right_x, beam_y - partition_height),
             (right_x, beam_y + partition_height), (0, 255, 120), 3)
    
    cv2.line(frame, (left_boundary, beam_y - partition_height),
             (left_boundary, beam_y + partition_height), (0, 255, 120), 3)
    cv2.line(frame, (right_boundary, beam_y - partition_height),
             (right_boundary, beam_y + partition_height), (0, 255, 120), 3)
    
    steering_history.append(lane_center)
    if len(steering_history) > 5:
        steering_history.pop(0)
    smooth_center = sum(steering_history) / len(steering_history)
    
    constrained_center = max(left_boundary, min(right_boundary, int(smooth_center)))
    
    cv2.circle(frame, (constrained_center, beam_y), 6, (0, 0, 255), -1)
    cv2.circle(frame, (constrained_center, beam_y), 10, (0, 0, 255), 2)
    
    text_size = cv2.getTextSize(advice, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = width // 2 - (text_size[0] // 2)
    text_y = beam_y - 40
    cv2.putText(frame, advice, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    deviation_text = f"Deviation: {abs(deviation)}px"
    if deviation != 0:
        deviation_text += " Left" if deviation < 0 else " Right"
    cv2.putText(frame, deviation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def process_yolo_detections(detected_labels):
    if 'stop' in detected_labels:
        queue_command('s')
        print('stop')
    elif 'right' in detected_labels:
        queue_command('r')
        print('right')
    elif 'left' in detected_labels:
        queue_command('l')
        print('left')
    elif 'go' in detected_labels:
        queue_command('f')
        print('straight')

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

        yolo_frame = frame.copy()
        results = model(frame, verbose=False)
        detected_labels = set()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                label = model.names[cls]
                detected_labels.add(label)
                cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(yolo_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('YOLO Detections', cv2.resize(yolo_frame, (720, 540)))

        process_yolo_detections(detected_labels)

        if not detected_labels:
            height, width = frame.shape[:2]
            roi_vertices = np.array([[
                (0, height // 2),
                (width, height // 2),
                (width, height),
                (0, height)
            ]], dtype=np.int32)

            frame = draw_roi(frame.copy(), roi_vertices)
            edges = preprocess_frame(frame, ui)
            roi_edges = apply_roi_mask(edges, roi_vertices)
            vertical_lines = detect_vertical_lines(roi_edges, ui)
            extrapolated_lines = average_and_extrapolate_lines(frame, vertical_lines)
            frame_with_lanes = draw_lanes(frame.copy(), extrapolated_lines)
            lane_center, advice, deviation = calculate_steering(frame, extrapolated_lines)
            final_frame = draw_steering_guidance(frame_with_lanes, lane_center, advice, deviation)
            cv2.imshow('Lane Detection System', cv2.resize(final_frame, (720, 540)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()