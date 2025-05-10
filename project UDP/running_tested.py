import cv2
import numpy as np
import requests
import time
import http.client

bot_num = 82 
host = f"192.168.{bot_num}.10" 
stream_url = f"http://192.168.{bot_num}.14:81/stream" 
port = 80  
previous_time = 0
last_command_time = 0

def send_command(host, port, command):
    global last_command_time  # Access the global variable

    current_time = time.time()
    time_elapsed = current_time - last_command_time  # Calculate time since last command

    if time_elapsed < 8:
        print(f"Time passed since last command: {time_elapsed:.2f} seconds")
        return None  # Don't send command if 8 seconds haven't passed

    for attempt in range(10):
        conn = None
        try:
            path = f"/?cmd={command}"
            conn = http.client.HTTPConnection(host, port, timeout=1)
            conn.request("GET", path)
            response = conn.getresponse()
            response.read()
            last_command_time = time.time()  # Update last command time
            print(f"Command '{command}' sent successfully!")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        finally:
            if conn:
                conn.close()
    
    return None

def calculate_steering_angle(frame, largest_contour):
    height, width = frame.shape[:2]
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else width // 2
    center_offset = cx - (width // 2)
    steering_angle = -float(center_offset) * 0.1
    return steering_angle, cx

def apply_steering_control(steering_angle):

    if abs(steering_angle) < 10:
        send_command(host, port, "f(400)")
        return "Go Straight"
    elif steering_angle > 0:
        send_command(host, port, "l(100)")
        return "Turn Left"
    else:
        send_command(host, port, "r(100)")
        return "Turn Right"

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
                        yield frame
                    
    except Exception as e:
        print(f"Stream error: {e}")
        return None

# Example usage:

for frame in stream_video(stream_url):
    if frame is None:
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

        # Draw guiding bullet point
        cv2.circle(highlighted_frame, (center_x, height - 50), 5, (0, 0, 255), -1)
        cv2.line(highlighted_frame, (width // 2, height - 50), (center_x, height - 50), (0, 0, 255), 2)
        
        # Draw steering direction above the bullet point
        text_size = cv2.getTextSize(steering_direction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = height - 80
        cv2.putText(highlighted_frame, steering_direction, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        send_command(host, port, "s")
        highlighted_frame = frame

    cv2.polylines(highlighted_frame, [roi_vertices], True, (255, 0, 0), 2)
    cv2.imshow('Lane Detection with Steering', cv2.resize(highlighted_frame, (720, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()