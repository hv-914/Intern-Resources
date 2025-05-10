import cv2
import numpy as np
import requests

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

def access_camera_stream():
    # URL of the ESP32-CAM stream
    stream_url = "http://192.168.82.14:81/stream"
    
    # Access the video stream
    for frame in stream_video(stream_url):
        # Display the frame
        cv2.imshow('ESP32-CAM Stream', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    access_camera_stream()