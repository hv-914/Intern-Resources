import cv2
import numpy as np
import time
import http.client
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommandConnectionManager:
    def __init__(self, host, port, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connections = {
            'forward': self._create_connection(),
            'left': self._create_connection(),
            'right': self._create_connection()
        }
    
    def _create_connection(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
    
    def _refresh_connection(self, command_type):
        try:
            self.connections[command_type].close()
        except:
            pass
        self.connections[command_type] = self._create_connection()
    
    def send_command(self, command_type):
        commands = {
            'forward': '/?cmd=f',
            'left': '/?cmd=l(500)',
            'right': '/?cmd=r(500)'
        }
        
        try:
            conn = self.connections[command_type]
            conn.request("GET", commands[command_type])
            response = conn.getresponse()
            response.read()
            return True
        except Exception as e:
            logger.error(f"Command error: {e}")
            self._refresh_connection(command_type)
            return False

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
            conn_manager.send_command('forward')
            last_direction = "Go Straight"
        elif steering_angle > 0:
            conn_manager.send_command('left')
            last_direction = "Turn Left"
        else:
            conn_manager.send_command('right')
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
    
    conn_manager = CommandConnectionManager(control_host, 80)
    last_command_time = time.time()
    command_cooldown = 1
    last_direction = "Go Straight"
    
    session = requests.Session()
    
    try:
        while True:
            try:
                response = session.get(stream_url, stream=True, timeout=5)
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
                            
                            if frame is None:
                                continue
                            
                            frame, last_command_time, last_direction = process_frame(
                                frame, conn_manager, last_command_time, 
                                command_cooldown, last_direction
                            )
                            
                            cv2.imshow('Lane Detection with Steering', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                return
            except Exception as e:
                logger.error(f"Stream error: {e}")
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    finally:
        cv2.destroyAllWindows()
        for conn in conn_manager.connections.values():
            try:
                conn.close()
            except:
                pass

if __name__ == "__main__":
    main()