import cv2
import tensorflow as tf
import numpy as np

# Load the YOLOv8 model
model = YOLO(r'C:\Users\merit\PROJECTS\best.pt')

# Load your image generator model
model = tf.keras.models.load_model("path_to_your_model.h5")

# Function to preprocess webcam frame before feeding into the model
def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)  # Resize to match model input size
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Function to postprocess model output
def postprocess_output(output):
    output = np.squeeze(output)  # Remove batch dimension
    output = (output * 255).astype(np.uint8)  # Denormalize to [0, 255]
    return output

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Process the frame
    target_size = (128, 128)  # Replace with your model's input size
    processed_frame = preprocess_frame(frame, target_size)
    
    # Generate output using the model
    output = model.predict(processed_frame)
    generated_image = postprocess_output(output)

    # Show original and generated image side by side
    combined_frame = cv2.hconcat([frame, cv2.resize(generated_image, frame.shape[1::-1])])
    cv2.imshow("Webcam - Original and Generated", combined_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
