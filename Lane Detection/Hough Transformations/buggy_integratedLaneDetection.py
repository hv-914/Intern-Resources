import cv2
import numpy as np
from datetime import datetime
import time
import http.client
import requests
from collections import deque
from threading import Thread, Event
import logging

# Global variables
left_lane_history = []
right_lane_history = []
steering_history = []
bot_num = input("Enter bot number : ")
host = "192.168."+bot_num+".10"
port = 80
running = Event()  # Use Event for better thread control
stream_url = 'http://192.168.99.14:81/stream'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamBuffer:
    def __init__(self, maxsize=10):
        self.buffer = deque(maxlen=maxsize)
        self.latest_frame = None
    
    def add_frame(self, frame):
        self.buffer.append(frame)
        self.latest_frame = frame
    
    def get_latest_frame(self):
        return self.latest_frame

def create_session():
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=3,
        pool_connections=1,
        pool_maxsize=1,
        pool_block=False
    )
    session.mount('http://', adapter)
    return session

def stream_video():
    running.set()  # Set the event flag
    stream_buffer = StreamBuffer()
    session = create_session()
    
    # Connection parameters
    connection_timeout = 5
    read_timeout = 10
    retry_delay = 1
    max_retries = 3
    
    while running.is_set():
        retry_count = 0
        while retry_count < max_retries and running.is_set():
            try:
                response = session.get(
                    stream_url,
                    stream=True,
                    timeout=(connection_timeout, read_timeout)
                )
                
                if response.status_code == 200:
                    bytes_data = bytes()
                    for chunk in response.iter_content(chunk_size=8192):  # Increased buffer size
                        if not running.is_set():
                            break
                            
                        bytes_data += chunk
                        while True:
                            a = bytes_data.find(b'\xff\xd8')
                            b = bytes_data.find(b'\xff\xd9')
                            
                            if a != -1 and b != -1:
                                jpg = bytes_data[a:b+2]
                                bytes_data = bytes_data[b+2:]
                                
                                try:
                                    img = cv2.imdecode(
                                        np.frombuffer(jpg, dtype=np.uint8),
                                        cv2.IMREAD_COLOR
                                    )
                                    
                                    if img is not None:
                                        stream_buffer.add_frame(img)
                                        process_and_display_frame(img)
                                    
                                except cv2.error as e:
                                    logger.warning(f"Failed to decode frame: {e}")
                                    continue
                            else:
                                break
                                
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            running.clear()
                            break
                            
                else:
                    logger.error(f"Bad response status: {response.status_code}")
                    time.sleep(retry_delay)
                    retry_count += 1
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Connection error: {e}")
                time.sleep(retry_delay)
                retry_count += 1
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(retry_delay)
                retry_count += 1
                continue
    
    # Cleanup
    session.close()
    cv2.destroyAllWindows()

def process_and_display_frame(frame):
    try:
        height, width = frame.shape[:2]
        roi_vertices = np.array([[
            (0, height // 2),
            (width, height // 2),
            (width, height),
            (0, height)
        ]], dtype=np.int32)
        
        # Create UI instance only once if needed
        ui = LaneDetectionUI()
        
        # Process frame
        frame_with_roi = draw_roi(frame.copy(), roi_vertices)
        edges = preprocess_frame(frame_with_roi, ui)
        roi_edges = apply_roi_mask(edges, roi_vertices)
        vertical_lines = detect_vertical_lines(roi_edges, ui)
        extrapolated_lines = average_and_extrapolate_lines(frame, vertical_lines)
        
        # Draw lanes and guidance
        frame_with_lanes = draw_lanes(frame.copy(), extrapolated_lines)
        lane_center, advice, deviation = calculate_steering(frame, extrapolated_lines)
        final_frame = draw_steering_guidance(frame_with_lanes, lane_center, advice, deviation)
        
        # Show frame
        cv2.imshow('Lane Detection System', final_frame)
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")

def send_re_httpquest(host, port, path):
    """Send HTTP request with retries."""
    retries = 10  # Number of retries
    for _ in range(retries):
        conn = None
        try:
            conn = http.client.HTTPConnection(host, port)
            conn.request("GET", path)
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
            if _ < retries - 1:
                time.sleep(1)
        finally:
            if conn:
                conn.close()
    return None

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

def draw_roi(frame, vertices):
    """Draw ROI as a semi-transparent blue rectangle"""
    overlay = frame.copy()
    cv2.fillPoly(overlay, vertices, (255, 100, 0))  # Blue color (BGR)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # 20% opacity
    cv2.polylines(frame, vertices, True, (255, 100, 0), 2)  # Draw border
    return frame

def preprocess_frame(frame, ui):
    """Basic frame preprocessing"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, ui.threshold_values['canny_low'], 
                     ui.threshold_values['canny_high'])
    return edges

def apply_roi_mask(edges, vertices):
    """Apply a mask to focus on the region of interest"""
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def detect_vertical_lines(edges, ui):
    """Detect vertical lines using Hough transform"""
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
    """Average and extrapolate the detected lines"""
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
    """Draw detected lane lines"""
    if lanes[0]:
        cv2.line(frame, (lanes[0][0], lanes[0][1]), (lanes[0][2], lanes[0][3]), (0, 255, 255), 5)
    if lanes[1]:
        cv2.line(frame, (lanes[1][0], lanes[1][1]), (lanes[1][2], lanes[1][3]), (0, 255, 255), 5)
    return frame

def calculate_steering(frame, lanes):
    """Calculate steering position and advice with deviation"""
    height, width = frame.shape[:2]
    frame_center = width // 2

    if not lanes or (lanes[0] is None and lanes[1] is None):
        return frame_center, "No Lane Detected", 0

    left_lane = lanes[0]
    right_lane = lanes[1]

    if left_lane and right_lane:
        left_x = (left_lane[0] + left_lane[2]) // 2
        right_x = (right_lane[0] + right_lane[2]) // 2
        lane_center = (left_x + right_x) // 2
    elif left_lane:
        lane_center = (left_lane[0] + left_lane[2]) // 2
    elif right_lane:
        lane_center = (right_lane[0] + right_lane[2]) // 2
    else:
        return frame_center, "No Lane Detected", 0

    deviation = lane_center - frame_center
    
    if deviation < -25:
        temp = abs(deviation + 25)
        temp = (temp//10) + 1
        advice = f"Turn Left ({temp})"
        print(f'l{temp}')
        path = f"/?cmd=l(200)"
        send_re_httpquest(host, port, path)

    elif deviation > 25:
        temp = deviation - 25
        temp = (temp//10) + 1
        advice = f"Turn Right ({temp})"
        print(f'r{temp}')
        path = f"/?cmd=r(200)"
        send_re_httpquest(host, port, path)

    else:
        print('0')
        advice = "Go Straight"
        path = f"/?cmd=f"
        send_re_httpquest(host, port, path)

    return lane_center, advice, deviation

def draw_steering_guidance(frame, lane_center, advice, deviation):
    """Draw steering guidance with I-beam visualization and deviation"""
    height, width = frame.shape[:2]
    beam_y = int(height * 0.85)
    beam_width = 200
    
    # Calculate beam boundaries
    left_boundary = width//2 - beam_width//2
    right_boundary = width//2 + beam_width//2
    
    # Draw main guidance beam
    cv2.line(frame, (left_boundary, beam_y), 
             (right_boundary, beam_y), (0, 255, 120), 3)
    
    # Draw partition lines
    left_x = width//2 - beam_width//8
    right_x = width//2 + beam_width//8
    
    partition_height = 12
    cv2.line(frame, (left_x, beam_y - partition_height),
             (left_x, beam_y + partition_height), (0, 255, 120), 3)
    cv2.line(frame, (right_x, beam_y - partition_height),
             (right_x, beam_y + partition_height), (0, 255, 120), 3)
    
    # Draw endpoint vertical lines
    cv2.line(frame, (left_boundary, beam_y - partition_height),
             (left_boundary, beam_y + partition_height), (0, 255, 120), 3)
    cv2.line(frame, (right_boundary, beam_y - partition_height),
             (right_boundary, beam_y + partition_height), (0, 255, 120), 3)
    
    # Draw steering indicator with smoothing and boundary constraints
    steering_history.append(lane_center)
    if len(steering_history) > 5:
        steering_history.pop(0)
    smooth_center = sum(steering_history) / len(steering_history)
    
    # Constrain indicator position within beam boundaries
    constrained_center = max(left_boundary, min(right_boundary, int(smooth_center)))
    
    # Draw indicator circles
    cv2.circle(frame, (constrained_center, beam_y), 6, (0, 0, 255), -1)
    cv2.circle(frame, (constrained_center, beam_y), 10, (0, 0, 255), 2)
    
    # Draw advice text above the I-beam
    text_size = cv2.getTextSize(advice, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = width // 2 - (text_size[0] // 2)
    text_y = beam_y - 40
    cv2.putText(frame, advice, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw deviation text in the top-left corner
    deviation_text = f"Deviation: {abs(deviation)}px"
    if deviation != 0:
        deviation_text += " Left" if deviation < 0 else " Right"
    cv2.putText(frame, deviation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def main():
    stream_video()

if __name__ == "__main__":
    try:
        stream_thread = Thread(target=stream_video)
        stream_thread.start()
        
        # Main loop
        while running.is_set():
            time.sleep(0.1)  # Reduce CPU usage
            
    except KeyboardInterrupt:
        print("\nStopping stream...")
        running.clear()
        stream_thread.join()
        cv2.destroyAllWindows()
