from ultralytics import YOLO

model = YOLO("/home/peanut/Downloads/runs/detect/train/weights/best.pt")  

results = model.train(
    data="/home/peanut/Downloads/data.yaml",  # Update this with your dataset path
    epochs=500,  # Number of training epochs
    batch=16,  # Adjust based on GPU memory
    imgsz=640,  # Image size
    device="cuda"  # Use "cpu" if no GPU is available
)
