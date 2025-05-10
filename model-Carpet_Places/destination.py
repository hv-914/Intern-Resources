import cv2
import numpy as np
import socket
import time
import requests
from ultralytics import YOLO

# ESP32 Configuration
camera_ip = "192.168.82.14"  # ESP32-CAM IP
robot_ip = "192.168.82.10"   # ESP32 Robot IP
udp_port = 8888
stream_url = f"http://{camera_ip}:81/stream"

# Time control for commands
last_command_time = 0
COMMAND_DELAY = 1.5

# Load YOLO model
model = YOLO("C:/users/merit/PROJECTS/hola.pt")  # Path to your trained model
valid_destinations = ['School', 'Beach', 'House', 'Office']

def get_destination():
    while True:
        print("\nAvailable destinations:", ", ".join(valid_destinations))
        destination = input("Enter your destination: ").strip().title()
        if destination in valid_destinations:
            return destination
        print("Invalid destination. Please choose from the available options.")

def stream_video(stream_url):
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
    
    current_time = time.time()
    if current_time - last_command_time < COMMAND_DELAY:
        return "Command skipped - too soon"

    if command not in ['f', 'b', 'l', 'r', 's']:
        return "Invalid command! Use 'f', 'b', 'l', 'r', 's'."

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(command.encode(), (robot_ip, udp_port))
            response, _ = sock.recvfrom(1024)
            last_command_time = current_time
            return response.decode()
    except Exception as e:
        return f"Error: {e}"

def calculate_steering_angle(frame, largest_contour):
    height, width = frame.shape[:2]
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else width // 2
    center_offset = cx - (width // 2)
    steering_angle = -float(center_offset) * 0.1
    return steering_angle, cx

def apply_steering_control(steering_angle):
    if abs(steering_angle) < 8:
        return "Go Straight", send_udp_command('f')
    elif steering_angle > 0:
        return "Turn Left", send_udp_command('l')
    else:
        return "Turn Right", send_udp_command('r')

def process_object_detection(frame, current_destination):
    detection_frame = frame.copy()
    results = model(frame)[0]
    destination_found = False
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = model.names[int(class_id)]
            
            # Draw bounding box
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detection_frame, f"{label} {score:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if label == current_destination:
                destination_found = True
                
    return detection_frame, destination_found

# Main loop
current_destination = get_destination()

for frame in stream_video(stream_url):
    # Object Detection Window
    detection_frame, destination_found = process_object_detection(frame, current_destination)
    cv2.imshow('Object Detection', cv2.resize(detection_frame, (720, 540)))
    
    # Lane Detection Window
    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 106, 255, cv2.THRESH_BINARY_INV)

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

    if contours and not destination_found:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(overlay, [largest_contour], (0, 255, 0))
        highlighted_frame = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
        cv2.drawContours(highlighted_frame, [largest_contour], -1, (0, 255, 255), 2)

        steering_angle, center_x = calculate_steering_angle(frame, largest_contour)
        steering_direction, command_response = apply_steering_control(steering_angle)

        cv2.circle(highlighted_frame, (center_x, height - 50), 5, (0, 0, 255), -1)
        cv2.line(highlighted_frame, (width // 2, height - 50), (center_x, height - 50), (0, 0, 255), 2)
        
        text_size = cv2.getTextSize(steering_direction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = height - 80
        cv2.putText(highlighted_frame, steering_direction, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        time_left = max(0, COMMAND_DELAY - (time.time() - last_command_time))
        timing_text = f"Next command in: {time_left:.1f}s"
        cv2.putText(highlighted_frame, timing_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        highlighted_frame = frame
        if destination_found:
            send_udp_command('s')  # Stop the bot
            cv2.putText(highlighted_frame, f"Destination '{current_destination}' reached!", 
                       (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Lane Detection with Steering', cv2.resize(highlighted_frame, (720, 540)))
            cv2.waitKey(2000)  # Display message for 2 seconds
            current_destination = get_destination()  # Get new destination
            continue

    cv2.polylines(highlighted_frame, [roi_vertices], True, (255, 0, 0), 2)
    cv2.putText(highlighted_frame, f"Current destination: {current_destination}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Lane Detection with Steering', cv2.resize(highlighted_frame, (720, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()