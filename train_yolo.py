import torch
from ultralytics import YOLO

# Fix for PyTorch 2.6+ (allow YOLOv8 internal classes to load)
torch.serialization.add_safe_globals([
    __import__('ultralytics').nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential
])

# Load YOLOv8 model (choose small model for faster training)
model = YOLO('yolov8n.pt')  # you can also use 'yolov8s.pt'

# Train the YOLO model
results = model.train(
    data='data_yolo/data.yaml',   # path to your YOLO dataset config
    epochs=50,                    # adjust as needed
    imgsz=640,                    # image size
    batch=16,                     # batch size (adjust to your RAM)
    name='garbage_yolo',          # experiment name
    project='Models',             # where to save weights
)

print("✅ YOLO training complete!")
print("Best model saved at: Models/garbage_yolo/weights/best.pt")
