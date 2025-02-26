# main.py

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from inference import inference 
from chat import generate_deskripsi, generate_pencegahan, generate_penanganan, class_info_dict
import os

# Folder tempat gambar produk disimpan
PRODUCT_IMAGE_FOLDER = os.path.join(os.getcwd(), "product image")

# Mapping penyakit ke rekomendasi produk (nama produk harus persis sesuai dengan key di product_info)
disease_product_map = {
    "Apple Scab Leaf": ["Dithane M-45", "Score 250 EC"],
    "Apple rust leaf": ["Score 250 EC", "Amistar Top"],
    "Bell_pepper leaf spot": ["Dithane M-45", "Bordeaux-Mixture_20wp"],
    "Corn Gray leaf spot": ["Score 250 EC", "Amistar Top"],
    "Corn leaf blight": ["Dithane M-45", "Tilt 250 EC"],
    "Corn rust leaf": ["Score 250 EC", "Amistar Top"],
    "Potato leaf early blight": ["Dithane M-45", "Score 250 EC"],
    "Potato leaf late blight": ["Ridomil Gold 480 SC", "Dithane M-45"],
    "Squash Powdery mildew leaf": ["Sulfur-80-1", "Bio Trichoderma"],
    "Tomato Early blight leaf": ["Dithane M-45", "Score 250 EC"],
    "Tomato Septoria leaf spot": ["Score 250 EC", "Amistar Top"],
    "Tomato leaf bacterial spot": ["Bordeaux-Mixture_20wp", "actara-25wg"],
    "Tomato leaf late blight": ["Ridomil Gold 480 SC", "Dithane M-45"],
    "Tomato leaf mosaic virus": ["Decis 25 WG", "Bio Trichoderma"],
    "Tomato leaf yellow virus": ["Decis 25 WG", "actara-25wg"],
    "Tomato mold leaf": ["Dithane M-45", "Tilt 250 EC"],
    "Tomato two spotted spider mites leaf": ["Abamectin 18 EC", "actara-25wg"],
    "Grape leaf black rot": ["Bordeaux-Mixture_20wp", "Dithane M-45"],
    "Antraknosa": ["Dithane M-45", "Score 250 EC"],
    "Daun-bercak-cokelat": ["Bordeaux-Mixture_20wp", "Dithane M-45"],
    "Gemini-virus": ["Decis 25 WG", "Bio Trichoderma"]
}

# Detail masing-masing produk: nama file gambar dan instruksi penggunaan
product_info = {
    "Dithane M-45": {
        "image": "Dithane M-45.png",
        "usage": "Aplikasikan secara foliar sesuai dosis petunjuk pada label, biasanya 2-3 liter per hektar, semprot secara menyeluruh pada daun yang terinfeksi."
    },
    "Score 250 EC": {
        "image": "Score 250 EC.png",
        "usage": "Campurkan sesuai dosis yang dianjurkan, aplikasikan secara merata pada seluruh area tanaman, ulangi setiap 7-10 hari."
    },
    "Amistar Top": {
        "image": "Amistar Top.png",
        "usage": "Gunakan sesuai dosis anjuran; aplikasikan dengan semprot, hindari aplikasi pada jam panas untuk mencegah kerusakan tanaman."
    },
    "Bordeaux-Mixture_20wp": {
        "image": "Bordeaux-Mixture_20wp.png",
        "usage": "Campurkan tembaga sulfat dan kapur sesuai rasio yang dianjurkan, aplikasikan sebagai semprotan kontak secara preventif."
    },
    "Tilt 250 EC": {
        "image": "Tilt 250 EC.png",
        "usage": "Aplikasikan dengan dosis yang disarankan, pastikan penyemprotan merata, ulangi aplikasi sesuai instruksi pada label."
    },
    "Ridomil Gold 480 SC": {
        "image": "Ridomil Gold 480 SC.png",
        "usage": "Aplikasikan melalui penyemprotan sesuai petunjuk pada label, terutama pada tahap awal gejala untuk hasil optimal."
    },
    "Sulfur-80-1": {
        "image": "Sulfur-80-1.png",
        "usage": "Aplikasikan secara foliar pada saat cuaca sejuk untuk menghindari iritasi tanaman, ikuti dosis yang tertera pada label."
    },
    "Bio Trichoderma": {
        "image": "Bio Trichoderma.png",
        "usage": "Aplikasikan sebagai biofungisida, bisa disiram ke tanah atau disemprot secara foliar untuk meningkatkan kekebalan tanaman."
    },
    "Decis 25 WG": {
        "image": "Decis 25 WG.png",
        "usage": "Aplikasikan sebagai insektisida untuk mengendalikan vektor; dosis sesuai label, hindari kontak langsung dengan manusia dan hewan."
    },
    "actara-25wg": {
        "image": "actara-25wg.png",
        "usage": "Aplikasikan sesuai dosis yang direkomendasikan, efektif untuk pengendalian hama kecil seperti thrips dan mites."
    },
    "Abamectin 18 EC": {
        "image": "Abamectin 18 EC.png",
        "usage": "Gunakan sebagai acaricide; semprotkan pada daun dan bagian tanaman yang terinfeksi, ikuti dosis anjuran pada label."
    }
}

def load_image_as_base64(filename):
    try:
        image_path = os.path.join(PRODUCT_IMAGE_FOLDER, filename)
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        return None

app = Flask(__name__)
os.environ["FLASK_RUN_EXTRA_FILES"] = "D:\\Skripsi Gaming"

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
        
        deskripsi = generate_deskripsi(info)
        pencegahan = generate_pencegahan(info)
        penanganan = generate_penanganan(info)
        
        # Buat rekomendasi produk untuk penyakit yang terdeteksi
        recommendations = {}
        for class_id in detected_classes:
            disease_name = class_names.get(int(class_id), "Unknown Disease")
            if disease_name in disease_product_map:
                products = []
                for prod_name in disease_product_map[disease_name]:
                    prod_details = product_info.get(prod_name)
                    if prod_details:
                        prod_image = load_image_as_base64(prod_details["image"])
                        products.append({
                            "product_name": prod_name,
                            "usage_instructions": prod_details["usage"],
                            "image_base64": prod_image
                        })
                recommendations[disease_name] = products
        
        # Encode hasil inferensi gambar
        _, buffer = cv2.imencode('.jpg', detected_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "detected_label": [class_names.get(int(cls), "Unknown") for cls in detected_classes],
            "deskripsi": deskripsi,
            "pencegahan": pencegahan,
            "penanganan": penanganan,
            "product_recommendations": recommendations,
            "inferenced_image": image_base64
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
