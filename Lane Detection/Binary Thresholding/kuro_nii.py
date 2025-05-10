import cv2
import numpy as np

def region_of_interest(img):
    """ Mask the bottom half of the frame """
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    
    # Define a polygon to mask the bottom half
    polygon = np.array([[
        (0, height // 2),  # Top-left
        (width, height // 2),  # Top-right
        (width, height),  # Bottom-right
        (0, height)  # Bottom-left
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_black_lanes(frame):
    """ Detect black lane markings in the ROI """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to detect dark lane lines
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Apply region of interest mask
    roi = region_of_interest(thresholded)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(roi, 50, 150)

    return edges

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for lane detection
        lanes = detect_black_lanes(frame)

        # Display the original frame with the detected lane overlay
        cv2.imshow("Lane Detection", lanes)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()