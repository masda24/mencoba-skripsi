from ultralytics import YOLO
import numpy as np

# Mapping nama kelas untuk hasil yang lebih jelas
class_mapping = {
    "Whitefly": "Whitefly Chili Leaf",
    "Sehat": "Healthy Chili Leaf",
    "Kuning": "Yellow Chili Leaf",
    "Keriting": "Curly Chili Leaf",
    "Bercak": "Chili Leaf Spot",
    "Bell_pepper leaf": "Healthy Chili Leaf",
    "Bell_pepper leaf spot": "Chili Leaf Spot"
}

# Load model YOLO sekali saat aplikasi dimulai
model = YOLO('best.pt')

def inference(image):
    if image is None:
        raise ValueError("Error: No image provided")

    results = model(image, conf=0.3)
    infer = np.zeros(image.shape, dtype=np.uint8)
    detected_classes = []

    for r in results:
        infer = r.plot()  # Gambar hasil deteksi
        detected_classes = [r.names[int(cls)] for cls in r.boxes.cls.tolist()]  # Ambil nama kelas
        
        # Jika nama kelas tidak ada di class_mapping, pakai nama asli dari YOLO
        detected_classes = [class_mapping.get(name, name) for name in detected_classes]

    return infer, r.names, detected_classes
