import cv2
import numpy as np
import socket
import time

ip = "192.168.82.10"  # ESP32 Server IP
udp_port = 8888 

def send_udp_command(command):
    if command not in ['f', 'b', 'l', 'r', 's']:
        return "Invalid command! Use 'f', 'b', 'l', 'r', 's'."

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(command.encode(), (ip, udp_port))  # Send command
            response, _ = sock.recvfrom(1024)  # Receive response
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
    if abs(steering_angle) < 10:
        send_udp_command('f')
        return "Go Straight"
    elif steering_angle > 0:
        send_udp_command('l')
        return "Turn Left"
    else:
        send_udp_command('r')
        return "Turn Right"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
        highlighted_frame = frame

    cv2.polylines(highlighted_frame, [roi_vertices], True, (255, 0, 0), 2)
    cv2.imshow('Lane Detection with Steering', cv2.resize(highlighted_frame, (720, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()