import cv2
import numpy as np
import requests
from ultralytics import YOLO
import logging
import http.client

bot_num = 82 
host = f"192.168.{82}.10" 
port = 80 
# Suppress YOLO logging output
logging.getLogger("ultralytics").setLevel(logging.WARNING)

class RoboticCarNavigation:
    def __init__(self, ip_address):
        self.model = YOLO("C:/Users/merit/PROJECTS/obj/runs/detect/train2/weights/hola.pt", verbose=False)
        self.stream_url = f"http://{ip_address}:81/stream"
        self.locations = {'Beach', 'School', 'Office', 'House'}
        self.session = requests.Session()
    
    def send_command(self, host, port, command):
        for attempt in range(10):
            conn = None
            try:
                path = f"/?cmd={command}"
                conn = http.client.HTTPConnection(host, port, timeout=1)
                conn.request("GET", path)
                response = conn.getresponse()
                response.read()
                return True
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                #time.sleep(0.1)  # Reduced from 1s to 0.1s
            finally:
                if conn:
                    conn.close()
        return None

    def stream_video(self):
        bytes_data = bytes()
        
        try:
            response = self.session.get(self.stream_url, stream=True)
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

    def calculate_steering_angle(self, frame, largest_contour):
        height, width = frame.shape[:2]
        M = cv2.moments(largest_contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else width // 2
        center_offset = cx - (width // 2)
        steering_angle = -float(center_offset) * 0.1
        return steering_angle, cx

    def apply_steering_control(self, steering_angle):
        if abs(steering_angle) < 10:
            self.send_command(host, port, "f(1000)")
            return "Go Straight"
        elif steering_angle > 0:
            self.send_command(host, port, "f(1000)")
            return "Turn Left" 
        else:
            self.send_command(host, port, "f(1000)")
            return "Turn Right"

    def process_lane_detection(self, frame):
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

        if not contours:
            return frame, None, None

        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(overlay, [largest_contour], (0, 255, 0))
        highlighted_frame = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
        cv2.drawContours(highlighted_frame, [largest_contour], -1, (0, 255, 255), 2)

        steering_angle, center_x = self.calculate_steering_angle(frame, largest_contour)
        steering_direction = self.apply_steering_control(steering_angle)

        cv2.polylines(highlighted_frame, [roi_vertices], True, (255, 0, 0), 2)
        return highlighted_frame, steering_direction, center_x

    def detect_location(self, frame, target_location=None):
        results = self.model(frame, verbose=False)
        detected_objects = results[0].boxes.cls
        detected_labels = [self.model.names[int(cls)] for cls in detected_objects]
        
        if target_location and target_location in detected_labels:
            return True
        return False

    def navigate(self):
        try:
            # Get source location
            while True:
                source = input("Enter source location (Beach/School/Office/House): ").strip().title()
                if source in self.locations:
                    break
                print("Invalid location! Please choose from available locations.")

            # Verify source location
            print(f"\nVerifying you're at {source}...")
            for frame in self.stream_video():
                if frame is None:
                    continue

                if self.detect_location(frame, source):
                    print(f"Confirmed: Starting from {source}")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

            # Get destination
            while True:
                destination = input("Enter destination (Beach/School/Office/House): ").strip().title()
                if destination in self.locations and destination != source:
                    break
                print("Invalid destination or same as source!")

            print(f"\nNavigating from {source} to {destination}")
            print("Press 'q' to stop navigation")

            for frame in self.stream_video():
                if frame is None:
                    continue

                # Process lane detection
                processed_frame, steering_direction, center_x = self.process_lane_detection(frame)
                
                if steering_direction:
                    height = frame.shape[0]
                    cv2.circle(processed_frame, (center_x, height - 50), 5, (0, 0, 255), -1)
                    cv2.line(processed_frame, (frame.shape[1] // 2, height - 50), 
                            (center_x, height - 50), (0, 0, 255), 2)
                    
                    text_size = cv2.getTextSize(steering_direction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = height - 80
                    cv2.putText(processed_frame, steering_direction, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Check if destination reached
                if self.detect_location(processed_frame, destination):
                    print(f"\nDestination {destination} reached!")
                    break

                cv2.imshow("Navigation", cv2.resize(processed_frame, (720, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nNavigation cancelled.")
                    break

        finally:
            cv2.destroyAllWindows()

def main():
    ip_address = "192.168.82.14"  # Replace with your camera's IP address
    navigator = RoboticCarNavigation(ip_address)
    navigator.navigate()

if __name__ == "__main__":
    main()