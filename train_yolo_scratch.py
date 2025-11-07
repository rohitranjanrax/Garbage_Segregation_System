from ultralytics import YOLO

# Train YOLO from scratch (no pre-trained weights)
model = YOLO('yolov8n.yaml')  # you can also try yolov8s.yaml for slightly larger model

# Start training
model.train(
    data='data_yolo/data.yaml',  # path to your dataset yaml
    epochs=50,                   # number of training epochs (increase if needed)
    imgsz=640,                   # image size
    batch=8,                     # adjust based on your GPU/CPU memory
    name='garbage_yolo_scratch', # run name
)
