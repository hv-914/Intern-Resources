import cv2
import numpy as np
import socket
import time
import requests 

# ESP32 Configuration
camera_ip = "192.168.82.14"  # ESP32-CAM IP
robot_ip = "192.168.82.10"   # ESP32 Robot IP
udp_port = 8888
stream_url = f"http://{camera_ip}:81/stream"
speed = 150

# Initialize adaptive contour area tracking
contour_areas_history = []
MAX_HISTORY_SIZE = 10
MIN_CONTOUR_AREA_FACTOR = 0.5  # Start with 50% of the average

# Initialize FPS variables
prev_frame_time = 0
new_frame_time = 0

# Time control for commands
last_command_time = 0
COMMAND_DELAY = 0

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
                        # Rotate the frame 180 degrees to fix upside down image
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

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(command.encode(), (robot_ip, udp_port))  # Send command
        response, _ = sock.recvfrom(1024)  # Receive response
        return response.decode()
    
def calculate_steering_angle(frame, largest_contour):
    height, width = frame.shape[:2]
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else width // 2
    center_offset = cx - (width // 2)
    steering_angle = -float(center_offset) * 0.1
    return steering_angle, cx

def apply_steering_control(steering_angle):
    if abs(steering_angle) < 14:
        send_udp_command(f'f:{speed}')
        return "Go Straight" 
    elif steering_angle > 0:
        send_udp_command(f'l:{speed}')
        return "Turn Left" 
    else:
        send_udp_command(f'r:{speed}')
        return "Turn Right"

def get_adaptive_min_contour_area():
    """Calculate the adaptive minimum contour area based on history"""
    if not contour_areas_history:
        return 10000  # Default value if no history yet
    
    avg_area = sum(contour_areas_history) / len(contour_areas_history)
    return avg_area * MIN_CONTOUR_AREA_FACTOR

def main():
    prev_frame_time = time.time()
    for frame in stream_video(stream_url):
        height, width = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY_INV)
        
        roi_vertices = np.array([[(0, height // 2), (width, height // 2), (width, height), (0, height)]], dtype=np.int32)
        
        mask = np.zeros_like(binary_frame)
        cv2.fillPoly(mask, roi_vertices, 255)
        roi_result = cv2.bitwise_and(binary_frame, mask)
        
        contours, _ = cv2.findContours(roi_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = np.zeros_like(frame)
        
        # Initialize variables for when no lanes are detected
        lanes_detected = False
        highlighted_frame = frame.copy()
        
        # Update contour area history if we have contours
        if contours:
            # Get the largest contour area for history tracking
            largest_area = max([cv2.contourArea(cnt) for cnt in contours])
            contour_areas_history.append(largest_area)
            
            # Keep history at a reasonable size
            if len(contour_areas_history) > MAX_HISTORY_SIZE:
                contour_areas_history.pop(0)
        
        # Get the adaptive minimum contour area threshold
        MIN_CONTOUR_AREA = get_adaptive_min_contour_area()
        
        # Check if contours exist and the largest one is above our threshold
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        
        if valid_contours:
            lanes_detected = True
            largest_contour = max(valid_contours, key=cv2.contourArea)
            cv2.fillPoly(overlay, [largest_contour], (0, 255, 0))
            highlighted_frame = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
            cv2.drawContours(highlighted_frame, [largest_contour], -1, (0, 255, 255), 2)
            
            steering_angle, center_x = calculate_steering_angle(frame, largest_contour)
            steering_direction = apply_steering_control(steering_angle)
            
            # Draw guiding bullet point
            cv2.circle(highlighted_frame, (center_x, height - 50), 5, (0, 0, 255), -1)
            cv2.line(highlighted_frame, (width // 2, height - 50), (center_x, height - 50), (0, 0, 255), 2)
            
            # Draw steering direction above the bullet point
            text_size = cv2.getTextSize(steering_direction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = height - 80
            cv2.putText(highlighted_frame, steering_direction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # No lanes detected or all contours are below threshold
            send_udp_command('s')
            message = "No Lanes Detected"
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = width // 2 - text_size[0] // 2
            text_y = 2 * (height // 3)
            cv2.putText(highlighted_frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw ROI outline
        cv2.polylines(highlighted_frame, [roi_vertices], True, (255, 0, 0), 2)
        
        # Display the current adaptive threshold
        cv2.putText(highlighted_frame, f'Min Area: {int(MIN_CONTOUR_AREA)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate and display Frame Rates
        new_frame_time = time.time()
        fps = 0 if (new_frame_time-prev_frame_time == 0) else (1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time
        fps = int(fps) if fps > 0 else 0

        cv2.putText(highlighted_frame, f'FPS: {fps}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(binary_frame, f'FPS: {fps}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Display the frames
        cv2.imshow('Binary Frame', cv2.resize(binary_frame, (720, 540)))
        cv2.imshow('Lane Detection with Steering', cv2.resize(highlighted_frame, (720, 540)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    send_udp_command('s')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()