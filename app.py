from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import base64

from chat import generate_insight, class_info_dict
from inference import inference  # Menggunakan inference.py

app = FastAPI()

@app.post("/predict")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    detected_image, class_names, detected_classes = inference(img)
    
    info = ""
    for i, class_id in enumerate(detected_classes):
        disease_name = class_names.get(int(class_id), "Unknown Disease")
        disease_info = class_info_dict.get(disease_name, "No info available")
        info += f"{disease_name}: {disease_info}"
        if i < len(detected_classes) - 1:
            info += ", "
    
    insight = generate_insight(info)
    
    _, buffer = cv2.imencode('.jpg', detected_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "detected_label": [class_names.get(int(cls), "Unknown") for cls in detected_classes],
        "insight": insight,
        "inferenced_image": image_base64
    }
