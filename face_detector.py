#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
from PIL import Image
#----------------------------------------------------------------------------------------------------------- 데이터셋 설정 
# 캠 생성
cam = cv2.VideoCapture(0)
cam.set(3, 640) # width
cam.set(4, 480) # height
face_detector = cv2.CascadeClassifier('C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/haarcascades/haarcascade_eye.xml')

# 각 사용자를 구별하기 위한 고유의 id 생성
face_id = input('\n id를 입력 후 엔터를 눌러주세요 : ')
print("\n 카메라를 쳐다봐주세요.")

# 학습시킬 사진 30장 count
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)                                 # 캠 좌우 반전 여부 (-1 : 반전)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)           # 캠 흑백/컬러 여부(RGB 지정)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)   # 객체 검출 메소드 (얼굴)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) # 객체 검출 메소드 설정
        count += 1
        # dataset 폴더에 저장 & 출력
        cv2.imwrite("C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        
    cv2.imshow('image', img)
        
    # ESC 키로 종
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    
    # 30장의 사진을 모두 캡처 후 종료
    elif count >= 10:
         break
     
 #초기화
cam.release()
cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------- 학습
path = 'C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/dataset'      # path : dataset 경로
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/haarcascades/haarcascade_frontalface_default.xml");

# 이미지 및 데이터 가져오기
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # 흑백 변환
        img_numpy = np.array(PIL_img,'uint8')        # 이미지 파일로 배열 생성
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))               # 학습

# 학습된 데이터 저장
recognizer.write('C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/trainer.yml')
print("\n [INFO] 사용자 {0}의 정보를 저장중입니다. 잠시만 기다려주세요.".format(len(np.unique(ids))))
#----------------------------------------------------------------------------------------------------------- 인식
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/trainer.yml')
cascadePath = "C:/Users/thdtn/Desktop/Graduation-Work-main/Graduation-Work-main/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

locker = ""

while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    
    eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor= 1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # confidence : 정확도
        if (confidence < 50):
            locker = "Welcome!"
            color = (0, 255, 0)
            confidence = "  {0}%".format(round(100 - confidence))
        elif (confidence >= 50):
            locker = "Unknown"
            color = (0, 0, 255)
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(locker), (x+5,y-5), font, 1, color, 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, color, 1)
        
        if((x >= 400)):
            cv2.putText(img, "WARNING! WARNING!", (30, 30), font, 1, (255, 0, 0), 1)
        
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n 프로그램 종료 및 정리")
cam.release()
cv2.destroyAllWindows()


# In[6]:





# In[ ]:




