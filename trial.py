# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:07:03 2023

@author: sbtng
"""

import cv2
from models import Yolov4


secondG=0
minuteG=0
hourG=0
secondL=0
minuteL=0
hourL=0
flag=0

img = cv2.imread("img/war.jpg")
name="stop"
model = Yolov4(weight_path='yolov4-helmet-detection.weights',
               class_name_path='class_names/helmet_classes.txt')
model1= Yolov4(weight_path='yolov4.weights',
               class_name_path='class_names/coco_classes.txt')

cv2.startWindowThread()

# Lee el archivo de video MP4
#cap = cv2.VideoCapture(0)
modulo2 = cv2.VideoCapture('img/mod2e.mp4')
modulo1 = cv2.VideoCapture('img/mod1e.mp4')

while True:
    
    # Lee un cuadro del video
    ret, frame = modulo2.read()
    if not ret:
        break
    # Lee un cuadro del video
    ret1, frame1 = modulo1.read()
    if not ret1:
        break
    
    secondG = secondG+1
    
    if secondG==60:
        minuteG=minuteG+1
        secondG=0
    if minuteG==60:
        hourG=hourG+1
        minuteG=0
    if secondL==60:
        minuteL=minuteL+1
        secondL=0
    if minuteL==60:
        hourL=hourL+1
        minuteL=0

    # Detecta objetos en el cuadro utilizando YOLOv4
    pred = model.predict(frame)
    pred1 = model1.predict(frame1)

    # Muestra el cuadro con los objetos detectados
    cv2.imshow("mod2", model.output_img)
    cv2.imshow("mod1", model1.output_img)
    
    eye=model.textReturn(frame)
    count=0
    for i in eye:
        if i == "head":
            count=count+1
    eye1=model1.textReturn(frame1)
    
    for i in eye1:
        if i == "person":
            count=count+1
    
    if count>0:
        cv2.imshow("stop", img)
        flag=1
    if count==0:
        secondL = secondL+1
        if flag ==1:
            cv2.destroyWindow("stop")
            flag=0
            
        
    #name=model.textReturn(frame)

    print("Global Timer")
    print ("%.2d:%.2d:%.2d" % (hourG,minuteG,secondG)) 

    print("Local Timer")
    print ("%.2d:%.2d:%.2d" % (hourL,minuteL,secondL)) 

    

    # Espera por un tecla para salir del bucle
    if cv2.waitKey(1) == ord('q'):
        break

# Limpia la memoria y cierra todas las ventanas
modulo2.release()
cv2.destroyAllWindows()