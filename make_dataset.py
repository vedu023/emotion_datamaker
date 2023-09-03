 

import cv2, os
import torch
import datetime
import numpy as np
import pandas as pd
# import face_recognition as fr
from PIL import Image
from model import LitModel
from torchvision import transforms
from deepface import DeepFace
import uuid, warnings
 

warnings.filterwarnings('ignore')

if os.path.exists('emotion_log.xlsx'):
    file = pd.read_excel('emotion_log.xlsx')

global df
df = pd.DataFrame(columns=['ID', 'Emotion', 'Timestamp'])

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+ r'haarcascade_frontalface_default.xml')
classifier = torch.load(r'/home/simbha/project_X/emotion_datamaker/emotion_detect.pt')
classifier.eval()

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
])


cap = cv2.VideoCapture(-1)

with torch.no_grad():
    while True:
        _, frame = cap.read()
        a = 1
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        face_id = []
        IDs_list = {}
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            im = gray[y:y+h, x:x+w]
            roi_gray = np.stack((im, im, im)).transpose((1, 2, 0))
            roi_face = Image.fromarray(roi_gray)
            
            if len(face_id) == 0:
                face_id.append(roi_gray)
                arr = str(roi_gray)
                id = str(uuid.uuid5(uuid.NAMESPACE_DNS, arr))
                IDs_list[id] = f'id{a}'
                IDs = IDs_list[id]
                a += 1
            
            else:
                ids = 0
                temp = False
                try:
                    for i in range(len(face_id)):
                        if DeepFace.verify(roi_gray, face_id[i])['verified']:
                            temp = True
                            ids = i
                            break

                    if temp == True:
                        IDs = (IDs_list.values())[i]     # img -> id 
                    else:
                        face_id.append(roi_gray)
                        arr = str(roi_gray) 
                        id = str(uuid.uuid5(uuid.NAMESPACE_DNS, arr))
                        IDs_list[id] = f'id{a}'
                        IDs = IDs_list[id]
                        a += 1
                except:
                    pass

            img1 = data_transform(roi_face)
            img1 = img1[np.newaxis, :] 
            output = classifier(img1) 

            predication = torch.max(output.data, 1).indices
            predication = predication.item()
    
            label_position = (x,y)
            cv2.putText(frame, str(emotion_labels[predication]), label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            now = datetime.datetime.now()  

            df = pd.concat([df, pd.DataFrame({'ID': IDs, 'Emotion': str(emotion_labels[predication]), 'Timestamp': now}, index=[0])])
            df.to_excel('emotion_log.xlsx', index=False)
    
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()


 