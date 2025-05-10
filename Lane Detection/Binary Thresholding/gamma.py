import cv2
import numpy as np
import requests
from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from requests import Session

bot_num = input("Enter bot number : ")
host = "192.168."+bot_num+".10"
port = 80
command_queue = deque(maxlen=3)
last_command_time = datetime.now()
executor = ThreadPoolExecutor(max_workers=1)
session = Session()

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

def calculate_steering_angle(frame, largest_contour):
    height, width = frame.shape[:2]
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else width // 2
    center_offset = cx - (width // 2)
    steering_angle = -float(center_offset) * 0.1
    return steering_angle, cx

def apply_steering_control(steering_angle):
    if abs(steering_angle) < 2.5:
        queue_command('f')
        action = "Go Straight"
    elif steering_angle > 0:
        queue_command('l')
        action = "Turn Left"
    else:
        queue_command('r')
        action = "Turn Right"
    
    print(f"Action: {action}")  # Print the action to the terminal
    return action

while True:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if not ret:
        continue

    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 80, 255, cv2.THRESH_BINARY_INV)

    roi_vertices = np.array([[
        (0, height // 2),
        (width, height // 2),
        (width, height),
        (0, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(binary_frame)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_result = cv2.bitwise_and(binary_frame, mask)
    contours, _ = cv2.findContours(roi_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = np.zeros_like(frame)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(overlay, [largest_contour], (0, 255, 0))
        highlighted_frame = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
        cv2.drawContours(highlighted_frame, [largest_contour], -1, (0, 255, 255), 2)
 
        steering_angle, center_x = calculate_steering_angle(frame, largest_contour)
        steering_direction = apply_steering_control(steering_angle)

        cv2.circle(highlighted_frame, (center_x, height - 50), 5, (0, 0, 255), -1)

        cv2.line(highlighted_frame, (width // 2, height - 50), (center_x, height - 50), (0, 0, 255), 2)
        
        text_size = cv2.getTextSize(steering_direction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = height - 80
        cv2.putText(highlighted_frame, steering_direction, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        highlighted_frame = frame

    cv2.polylines(highlighted_frame, [roi_vertices], True, (255, 0, 0), 2)
    cv2.imshow('Binary Frame', binary_frame)
    cv2.imshow('Lane Detection with Steering', highlighted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
executor.shutdown()
