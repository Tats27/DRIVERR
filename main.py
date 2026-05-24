import cv2
import time
import urllib.request
import os
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================

if not os.path.exists("eye.xml"):
    print("Descargando configurador de ojos...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    urllib.request.urlretrieve(url, "eye.xml")

eye_cascade = cv2.CascadeClassifier("eye.xml")

print("Cargando modelo YOLOv8m...")
model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

# Variables de tiempo para el control de alertas
tiempo_ojos_cerrados = 0
tiempo_ojos_bloqueados = 0
tiempo_bostezo = 0

print("\n!!! SISTEMA INTEGRAL REPARADO: FLUIDEZ TOTAL ACTIVA !!!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.35, verbose=False)
    
    # ⚠️ IMPORTANTE: Cada cuadro de video inicia en NORMAL por defecto
    estado_alerta = "Estatus: Conduciendo Normal"
    color_alerta = (0, 255, 0)  # Verde
    
    detecto_persona = False
    detecto_bostezo_boca = False

    # 2. DETECCIONES DE OBJETOS DE YOLOv8
    for result in results:
        for box in result.boxes:
            nombre_objeto = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Alertas críticas de YOLO
            if nombre_objeto in ["cell phone", "telephone"]:
                estado_alerta = "!!! ALERTA: DISTRACCIÓN POR CELULAR !!!"
                color_alerta = (0, 0, 255)
            elif nombre_objeto == "lighter":
                estado_alerta = "!!! ALERTA ROJA: PROHIBIDO FUMAR !!!"
                color_alerta = (0, 0, 255)
            elif nombre_objeto in ["cup", "bottle", "glass"]:
                estado_alerta = "WARN: Conductor tomando bebida (Fatiga)"
                color_alerta = (255, 165, 0)
            
            # Detector de Persona
            elif nombre_objeto == "person":
                detecto_persona = True
                
            # detector de Boca/Bostezo directo de YOLO
            elif nombre_objeto == "mouth":
                alto_boca = y2 - y1
                ancho_boca = x2 - x1
                # Si la boca se abre verticalmente de forma exagerada, es un bostezo
                if alto_boca > (ancho_boca * 0.7):
                    detecto_bostezo_boca = True

    # 3. LÓGICA DE BOSTEZO CON REINICIO INMEDIATO
    if detecto_bostezo_boca:
        if tiempo_bostezo == 0:
            tiempo_bostezo = time.time()
        
        # Si mantienes el bostezo por más de 0.8 segundos se activa la alerta naranja
        if time.time() - tiempo_bostezo > 0.8:
            if estado_alerta == "Estatus: Conduciendo Normal":
                estado_alerta = "!!! ADVERTENCIA: BOSTEZO / FATIGA DETECTADA !!!"
                color_alerta = (255, 165, 0)
    else:
        # En cuanto cierras la boca, el temporizador se borra y regresa a verde instantáneamente
        tiempo_bostezo = 0

    # 4. LÓGICA DE SUEÑO Y LENTES DE SOL
    if detecto_persona:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alto_f, ancho_f, _ = frame.shape
        roi_gray = gray[0:int(alto_f*0.6), 0:ancho_f]
        
        ojos = eye_cascade.detectMultiScale(roi_gray, 1.1, 7, minSize=(30, 30))
        
        for (ex, ey, ew, eh) in ojos:
            cv2.circle(frame, (ex + ew//2, ey + eh//2), 18, (255, 255, 0), 2)
            
        if len(ojos) == 0:
            if tiempo_ojos_cerrados == 0:
                tiempo_ojos_cerrados = time.time()
                tiempo_ojos_bloqueados = time.time()
            
            if time.time() - tiempo_ojos_cerrados > 1.3:
                estado_alerta = "!!! ALERTA CRÍTICA: CONDUCTOR DORMIDO !!!"
                color_alerta = (0, 0, 255)
                
            if time.time() - tiempo_ojos_bloqueados > 5.0:
                if estado_alerta == "Estatus: Conduciendo Normal":
                    estado_alerta = "WARN: Vista Bloqueada (¿Usa Lentes de Sol?)"
                    color_alerta = (255, 165, 0)
        else:
            tiempo_ojos_cerrados = 0
            tiempo_ojos_bloqueados = 0

    # 5. DIBUJAR INTERFAZ
    annotated_frame = results[0].plot() if 'results' in locals() else frame
    cv2.putText(annotated_frame, estado_alerta, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_alerta, 3)
    cv2.imshow("Sistema Integral Inteligente - TATS", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()