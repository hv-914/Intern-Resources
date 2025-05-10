import cv2
import numpy as np
import socket
import time
import requests
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("C:/users/merit/PROJECTS/hola.pt")
labels = ["School", "Beach", "Office", "House"]

destination = "School"  # Change this to desired destination
found_destination = False  # Flag to check if destination is reached

# ESP32 Configuration
camera_ip = "192.168.82.14"
robot_ip = "192.168.82.10"
udp_port = 8888
stream_url = f"http://{camera_ip}:81/stream"
COMMAND_DELAY = 1.5
last_command_time = 0

def stream_video():
    session = requests.Session()
    bytes_data = bytes()
    try:
        response = session.get(stream_url, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                        yield frame
    except Exception as e:
        print(f"Stream error: {e}")
        return None

def send_udp_command(command):
    global last_command_time
    if time.time() - last_command_time < COMMAND_DELAY:
        return "Command skipped - too soon"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(command.encode(), (robot_ip, udp_port))
            last_command_time = time.time()
            return "Command sent"
    except Exception as e:
        return f"Error: {e}"

def detect_location(frame):
    global found_destination
    results = model(frame)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence > 0.5 and labels[class_id] == destination:
                x, y, w, h = map(int, box.xywh[0])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, labels[class_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                found_destination = True
    return frame

def calculate_steering_angle(frame, largest_contour):
    height, width = frame.shape[:2]
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else width // 2
    return -float(cx - (width // 2)) * 0.1, cx

def apply_steering_control(steering_angle):
    if found_destination:
        return "Stop", send_udp_command('s')
    if abs(steering_angle) < 8:
        return "Go Straight", send_udp_command('f')
    elif steering_angle > 0:
        return "Turn Left", send_udp_command('l')
    else:
        return "Turn Right", send_udp_command('r')

# Main loop
for frame in stream_video():
    frame = detect_location(frame)
    if not found_destination:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 106, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            steering_angle, center_x = calculate_steering_angle(frame, largest_contour)
            steering_direction, _ = apply_steering_control(steering_angle)
            cv2.circle(frame, (center_x, frame.shape[0] - 50), 5, (0, 0, 255), -1)
            cv2.putText(frame, steering_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        send_udp_command('s')
        cv2.putText(frame, "Destination Reached", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Detection & Lane Following', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
