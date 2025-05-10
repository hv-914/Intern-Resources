import cv2
import numpy as np
import requests
import time

# Global control variable
running = True  
stream_url = "http://192.168.99.14:81/stream"  # Replace with actual stream URL

def stream_video():
    global running
    session = requests.Session()

    while running:
        try:
            response = session.get(stream_url, stream=True)
            if response.status_code == 200:
                bytes_data = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    if not running:
                        break
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')  # Start of JPEG
                    b = bytes_data.find(b'\xff\xd9')  # End of JPEG
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]

                        # Convert to OpenCV image
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            
                            # Display the frame
                            cv2.imshow("Video Stream", img)

                            # Exit if 'q' is pressed
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                running = False
                                break

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)  # Wait before retrying

    # Cleanup
    cv2.destroyAllWindows()

# Run the function
stream_video()
