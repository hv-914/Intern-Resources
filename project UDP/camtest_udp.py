import socket
import numpy as np
import cv2

# Set up the UDP connection
UDP_IP = "192.168.82.14"  # IP of your ESP32-CAM
UDP_PORT = 12345          # Port your ESP32-CAM is sending the stream to

# Create a socket to listen to the incoming stream
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Create a buffer to store incoming data
frame_data = b""

while True:
    # Receive data from the ESP32-CAM
    packet, addr = sock.recvfrom(65536)  # buffer size is 65536 bytes

    # Append the incoming packet to the frame data buffer
    frame_data += packet

    # Look for the JPEG frame boundary in the buffer
    start_index = frame_data.find(b'\xff\xd8')  # JPEG start
    end_index = frame_data.find(b'\xff\xd9')  # JPEG end

    if start_index != -1 and end_index != -1:
        # Extract the JPEG image data
        jpeg_data = frame_data[start_index:end_index+2]

        # Convert the JPEG data into an image
        frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Display the image
        if frame is not None:
            cv2.imshow("ESP32-CAM Video Stream", frame)

        # Remove the processed frame from the buffer
        frame_data = frame_data[end_index+2:]

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the OpenCV window and close the socket
cv2.destroyAllWindows()
sock.close()
