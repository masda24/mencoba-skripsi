from ultralytics import YOLO
import numpy as np

def inference(image):
    if image is None:
        raise ValueError("Error: No image provided")

    model = YOLO('best.pt')
    results = model(image, conf=0.3)
    
    infer = np.zeros(image.shape, dtype=np.uint8)  
    classes = dict()
    namesInfer = []

    for r in results:
        infer = r.plot()  
        classes = r.names
        namesInfer = r.boxes.cls.tolist()

    return infer, classes, namesInfer


# image_path = r'D:\Temp Files\klasifikasi_penyakit_tanaman\Apple scab - mature infection on leaves.png'

# # Run inference
# inferred_image, class_names, detected_classes = inference(image_path)

# # Optionally, you can save the result or display it
# cv2.imshow("Inferred Image", inferred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
