import os
import shutil

# 🏷️ Class names — same as before
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 📂 Paths
input_dir = "Data/Val"  # or Data/Test if you named it that
output_images = "data_yolo/images/val"
output_labels = "data_yolo/labels/val"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

for class_id, class_name in enumerate(classes):
    folder_path = os.path.join(input_dir, class_name)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_img = os.path.join(folder_path, img_name)
            dst_img = os.path.join(output_images, img_name)
            shutil.copy(src_img, dst_img)

            label_path = os.path.join(output_labels, os.path.splitext(img_name)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("✅ Validation data conversion completed successfully!")
