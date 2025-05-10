import cv2
import torch
from ultralytics import YOLO

# Load YOLO model (change to your model path if needed)
model = YOLO("C:/Users/merit/PROJECTS/aiLite_turnSigns.pt")  # Replace with your custom model if applicable

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference (suppress logs)
    results = model(frame, verbose=False)  # Disable detailed logging

    detected_labels = set()  # Store unique labels

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls = int(box.cls[0].item())  # Class index
            label = model.names[cls]

            # Store unique label for terminal output
            detected_labels.add(label)

            # Draw rectangle and label (in popup window)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Print only the labels in the terminal
    if detected_labels:
        print(", ".join(detected_labels))  # Prints only unique labels

    # Show the frame
    cv2.imshow("YOLO Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()