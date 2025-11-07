import os

# 🏷️ Class names — same as your folders
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 📂 Paths
input_dir = "Data/Train"  # Path to your CNN dataset folders
output_images = "data_yolo/images/train"
output_labels = "data_yolo/labels/train"

# Create output directories if not exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Loop through all class folders
for class_id, class_name in enumerate(classes):
    folder_path = os.path.join(input_dir, class_name)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_img = os.path.join(folder_path, img_name)
            dst_img = os.path.join(output_images, img_name)

            # Copy image to YOLO image folder
            os.system(f'copy "{src_img}" "{dst_img}" >nul')

            # Create YOLO label file (whole image = bounding box)
            label_path = os.path.join(output_labels, os.path.splitext(img_name)[0] + '.txt')
            with open(label_path, 'w') as f:
                # Format: class_id center_x center_y width height (all normalized)
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("✅ Conversion completed successfully!")
print(f"Labels saved in: {output_labels}")
