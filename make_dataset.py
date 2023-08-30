 

import cv2
import torch
import datetime
import numpy as np
import pandas as pd
import face_recognition as fr
from model import LitModel


global df
df = pd.DataFrame(columns=['ID', 'Emotion', 'Timestamp'])

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+ r'haarcascade_frontalface_default.xml')
classifier = torch.load(r'/home/linescan/project_X/face_detection_dataset/emotion_detect.pt')
classifier.eval()

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(-1)

while True:
    _, frame = cap.read()
    IDs_list = {}
    a = 1
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]

        fcs = fr.face_encodings(roi_gray, num_jitters=5)[0]
        if fcs not in IDs_list.keys():
            IDs_list[fcs] = f'ID{a}'
        
        IDs = IDs_list[fcs]

        roi_gray = cv2.resize(roi_gray,(48, 48),interpolation=cv2.INTER_AREA)
        roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi = roi.astype('float')/255.0
        roi = np.expand_dims(roi,axis=0)

        prediction = classifier.predict(roi)[0]
        label=emotion_labels[prediction.argmax()]
        label_position = (x,y)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        now = datetime.datetime.now()  

        df = df.append({'ID': IDs, 'Emotion': label, 'Timestamp': now}, ignore_index=True)

    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df.to_excel('emotion_log.xlsx', index=False)
cap.release()
cv2.destroyAllWindows()