import pyautogui
import PySimpleGUI as sg
import numpy as np
import pandas as pd
from collections import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
from pandas.core.frame import DataFrame
import serial
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from imutils import paths
import numpy as np
import pickle
import datetime,gspread,random
import oauth2client
from oauth2client.service_account import ServiceAccountCredentials
import spread
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5

print("[INFO] loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())


print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)


def main():
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Text("Attendnace live", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
        
        
        [sg.Button("Exit", size=(10, 1))],
        
    ]

    # Create the window and show it without the plot
    window = sg.Window("Automatic attendnace", layout, location=(0,0), size=(900,600), keep_on_top=True )

    cap = cv2.VideoCapture(0)
    
    time.sleep(2.0)

    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
       
                
        _, frame = cap.read()
       
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > conf:

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        if fW < 20 or fH < 20:
                                continue

                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = le.classes_[j]
                        with open('student.csv', 'r') as csvFile:
                                reader = csv.reader(csvFile)
                                for row in reader:
                                        box = np.append(box, row)
                                        name = str(name)
                                        if name in row:
                                                person = str(row)
                                                print(name)
                                        listString = str(box)
                                        print(box)
                                        names =[]
                                        roll_no =[]
                                       
                                        
                                        if name in listString:
                                                singleList = list(flatten(box))
                                                listlen = len(singleList)
                                                Index = singleList.index(name)
                                                name = singleList[Index]
                                                Roll_Number = singleList[Index + 1]
                                            
                                                                 
                                text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
                                y = startY - 10 if startY - 10 > 10 else startY + 10
                                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                                (12, 140, 255), 2)
                                cv2.putText(frame, text, (startX, y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                now=datetime.datetime.now()                                      
                                d=now.strftime('%m/%d/%Y').replace('/0','/')
                                t=now.strftime('%H:%M:%S')
                                spread.enroll(name,Roll_Number,d,t)                                                 
                                pyautogui.alert(name)
                                pyautogui.pause = 5
                                time.sleep(2.0)   
                                
                

                    
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
       

    window.close()
main()

