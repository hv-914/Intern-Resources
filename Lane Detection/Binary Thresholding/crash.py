import cv2
import numpy as np
import time
import http.client
import requests
import socket

class ConnectionManager:
    def __init__(self, host, port):
        self.host = host
        self.port = port
    
    def send_request(self, path):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=1)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            return data
        except (socket.timeout, ConnectionError) as e:
            conn.close()
            return None
        except Exception as e:
            conn.close()
            print(f"Request error: {e}")
            return None

def process_frame(frame, conn_manager, last_command_time, command_cooldown, last_direction):
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
    
    if not contours:
        return frame, last_command_time, last_direction

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return frame, last_command_time, last_direction
        
    cx = int(M["m10"] / M["m00"])
    center_offset = cx - (width // 2)
    steering_angle = -float(center_offset) * 0.1

    current_time = time.time()
    if current_time - last_command_time >= command_cooldown:
        last_command_time = current_time
        if abs(steering_angle) < 10:
            conn_manager.send_request("/?cmd=f")
            last_direction = "Go Straight"
        elif steering_angle > 0:
            conn_manager.send_request("/?cmd=l(500)")
            last_direction = "Turn Left"
        else:
            conn_manager.send_request("/?cmd=r(500)")
            last_direction = "Turn Right"

    overlay = np.zeros_like(frame)
    cv2.fillPoly(overlay, [largest_contour], (0, 255, 0))
    frame = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)
    cv2.circle(frame, (cx, height - 50), 5, (0, 0, 255), -1)
    cv2.line(frame, (width // 2, height - 50), (cx, height - 50), (0, 0, 255), 2)
    
    text_size = cv2.getTextSize(last_direction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = cx - text_size[0] // 2
    text_y = height - 80
    cv2.putText(frame, last_direction, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.polylines(frame, [roi_vertices], True, (255, 0, 0), 2)
    return frame, last_command_time, last_direction

def main():
    bot_num = input("Enter bot number : ")
    control_host = f"192.168.{bot_num}.10"
    stream_url = f"http://192.168.{bot_num}.14:81/stream"
    
    conn_manager = ConnectionManager(control_host, 80)
    last_command_time = time.time()
    command_cooldown = 0.1
    last_direction = "Go Straight"
    
    session = requests.Session()
    
    try:
        last_time = time.time()
        
        while True:
            response = session.get(stream_url, stream=True)
            if response.status_code == 200:
                bytes_data = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                        frame, last_command_time, last_direction = process_frame(
                            frame, conn_manager, last_command_time, 
                            command_cooldown, last_direction
                        )

                        # FPS Calculation (Prevent division by zero)
                        current_time = time.time()
                        time_diff = current_time - last_time
                        if time_diff > 0:  # Avoid division by zero
                            fps = 1 / time_diff
                        else:
                            fps = 0  # Set FPS to 0 if time_diff is 0

                        last_time = current_time  # Update last time

                        # Display FPS on the frame
                        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

                        cv2.imshow('Lane Detection with Steering', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break  # Exit loop
                            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn_manager.send_request("/?cmd=s")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


