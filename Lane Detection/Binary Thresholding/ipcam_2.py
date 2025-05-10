import cv2
import numpy as np
import time
import http.client
import requests
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import socket
import logging
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self, host, port, pool_size=3, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connection_pool = Queue(maxsize=pool_size)
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
        
        # Initialize connection pool
        for _ in range(pool_size):
            try:
                self.connection_pool.put(self._create_connection())
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def _create_connection(self):
        try:
            conn = http.client.HTTPConnection(
                self.host, 
                self.port, 
                timeout=self.timeout
            )
            # Test the connection
            conn.connect()
            return conn
        except Exception as e:
            logger.error(f"Connection creation failed: {e}")
            raise
    
    def _get_connection(self):
        tries = 3
        while tries > 0:
            try:
                if not self.connection_pool.empty():
                    conn = self.connection_pool.get(timeout=1)
                    return conn
                return self._create_connection()
            except Exception as e:
                tries -= 1
                if tries == 0:
                    logger.error(f"Failed to get connection after 3 attempts: {e}")
                    raise
                time.sleep(0.5)
    
    def _return_connection(self, conn):
        try:
            if self.connection_pool.full():
                conn.close()
            else:
                # Test connection before returning to pool
                conn.sock.getpeername()  # Will raise error if connection is dead
                self.connection_pool.put(conn)
        except Exception:
            # If connection is dead, close it
            try:
                conn.close()
            except:
                pass
    
    def send_request(self, path):
        conn = None
        tries = 3
        while tries > 0:
            try:
                conn = self._get_connection()
                conn.request("GET", path)
                response = conn.getresponse()
                data = response.read().decode("utf-8")
                self._return_connection(conn)
                return data
            except (socket.timeout, ConnectionError, http.client.HTTPException) as e:
                tries -= 1
                logger.warning(f"Request attempt {3-tries} failed: {e}")
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                if tries == 0:
                    logger.error("Request failed after 3 attempts")
                    return None
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Unexpected error in send_request: {e}")
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                return None

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

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
            conn_manager.executor.submit(conn_manager.send_request, "/?cmd=f")
            last_direction = "Go Straight"
        elif steering_angle > 0:
            conn_manager.executor.submit(conn_manager.send_request, "/?cmd=l(500)")
            last_direction = "Turn Left"
        else:
            conn_manager.executor.submit(conn_manager.send_request, "/?cmd=r(500)")
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
    
    try:
        conn_manager = ConnectionManager(control_host, 80, timeout=5)
        last_command_time = time.time()
        command_cooldown = 0.1
        last_direction = "Go Straight"
        
        session = create_robust_session()
        
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
                            try:
                                jpg = bytes_data[a:b+2]
                                bytes_data = bytes_data[b+2:]
                                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                
                                if frame is None:
                                    logger.warning("Failed to decode frame")
                                    continue
                                
                                frame, last_command_time, last_direction = process_frame(
                                    frame, conn_manager, last_command_time, 
                                    command_cooldown, last_direction
                                )
                                
                                cv2.imshow('Lane Detection with Steering', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    return
                            except Exception as e:
                                logger.error(f"Frame processing error: {e}")
                                continue
                else:
                    logger.error(f"Stream connection failed with status code: {response.status_code}")
                    time.sleep(1)  # Wait before retrying
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Stream connection error: {e}")
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(1)  # Wait before retrying
                
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        cv2.destroyAllWindows()
        conn_manager.executor.shutdown()

if __name__ == "__main__":
    main()