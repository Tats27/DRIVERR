from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
import urllib.request

app = Flask(__name__)

if not os.path.exists("eye.xml"):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    urllib.request.urlretrieve(url, "eye.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")
model = YOLO("yolov8m.pt")

@app.route('/analizar', methods=['POST'])
def analizar_cuadro():
    try:
        file = request.files['image'].read()
        np_img = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "Error: Imagen inválida"})

        # Procesar con YOLOv8
        results = model(frame, conf=0.35, verbose=False)
        
        estado = "Conduciendo Normal"
        detecto_persona = False
        detecto_bostezo = False

        for result in results:
            for box in result.boxes:
                nombre_objeto = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if nombre_objeto in ["cell phone", "telephone"]:
                    return jsonify({"status": "DISTRACCIÓN POR CELULAR"})
                elif nombre_objeto in ["cup", "bottle", "glass"]:
                    estado = "CHOFER CON FATIGA"
                elif nombre_objeto == "person":
                    detecto_persona = True
                elif nombre_objeto == "mouth":
                    if (y2 - y1) > ((x2 - x1) * 0.7):
                        detecto_bostezo = True

        if detecto_bostezo and estado == "Conduciendo Normal":
            return jsonify({"status": "BOSTEZO DETECTADO"})

        if detecto_persona and estado == "Conduciendo Normal":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            alto, ancho, _ = frame.shape
            ojos = eye_cascade.detectMultiScale(gray[0:int(alto*0.6), 0:ancho], 1.1, 7, minSize=(30, 30))
            if len(ojos) == 0:
                return jsonify({"status": "CONDUCTOR DORMIDO"})

        return jsonify({"status": estado})
    
    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)