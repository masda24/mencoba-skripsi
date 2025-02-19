from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from inference import inference 
from chat import generate_insight, class_info_dict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        contents = file.read()
        np_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Jalankan inferensi
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
        
        return jsonify({
            "detected_label": [class_names.get(int(cls), "Unknown") for cls in detected_classes],
            "insight": insight,
            "inferenced_image": image_base64
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
