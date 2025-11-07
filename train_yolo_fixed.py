import torch
import torch.serialization
from ultralytics import YOLO
import ultralytics.nn.modules
import ultralytics.nn.tasks

# ✅ Allow YOLO classes and layers for PyTorch 2.6+
torch.serialization.add_safe_globals([
    torch.nn.modules.container.Sequential,
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.ReLU,
    torch.nn.SiLU,
    ultralytics.nn.modules.Conv,
    ultralytics.nn.tasks.DetectionModel
])

print("✅ PyTorch patched successfully — YOLO model loading enabled")

# ✅ Load YOLO model safely
try:
    print("⚙️ Loading YOLO weights safely (trusted file)...")
    model = YOLO("yolov8n.pt")  # Ensure this file exists in your folder
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    exit()

# ✅ Train the YOLO model
model.train(
    data="Data_yolo/data.yaml",  # Path to dataset YAML
    epochs=30,
    imgsz=640,
    device="cpu"  # CPU training since CUDA unavailable
)
