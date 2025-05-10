import cv2
from ultralytics import YOLO

# 🔹 Load your trained YOLOv8 model
model = YOLO("C:/users/merit/PROJECTS/hola.pt", verbose = False)  # Update with your trained model path

# 🔹 Open Webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change if multiple cameras are connected

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Run YOLOv8 inference on the frame
    results = model.predict(frame, conf=0.5)  # Set confidence threshold

    # 🔹 Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = f"{model.names[cls]}: {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 🔹 Display the frame
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔹 Release resources
cap.release()
cv2.destroyAllWindows()
