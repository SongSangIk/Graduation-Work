import cv2
import os
import numpy as np
from PIL import Image
#----------------------------------------------------------------------------------------------------------- 데이터셋 설정 
# 캠 생성
cam = cv2.VideoCapture(0)
cam.set(3, 640) # width
cam.set(4, 480) # height
face_detector = cv2.CascadeClassifier('C:/OpenCV/build/install/etc/haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:/OpenCV/build/install/etc/haarcascades/haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier('C:/OpenCV/build/install/etc/haarcascades/haarcascade_fullbody.xml')
upperCascade = cv2.CascadeClassifier('C:/OpenCV/build/install/etc/haarcascades/haarcascade_upperbody.xml')

# 각 사용자를 구별하기 위한 고유의 id 생성
face_id = input('\n id를 입력 후 엔터를 눌러주세요 : ')
# name = input("\n 이름을 입력해 주세요 : ")
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
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        
    # ESC 키로 종
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    
    # 30장의 사진을 모두 캡처 후 종료
    elif count >= 30:
         break
     
# 초기화
print("\n 프로그램 종료 및 정리")
cam.release()
cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------- 학습
path = 'C:/Users/user/dataset'      # path : dataset 경로
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("C:/OpenCV/build/install/etc/haarcascades/haarcascade_frontalface_default.xml");

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
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))               # 학습

# 학습된 데이터 저장
recognizer.write('C:/Users/user/trainer/trainer.yml')
print("\n [INFO] 사용자 {0}의 학습이 완료되었습니다. 프로그램을 종료합니다.".format(len(np.unique(ids))))
#----------------------------------------------------------------------------------------------------------- 인식
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/user/trainer/trainer.yml')
cascadePath = "C:/OpenCV/build/install/etc/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

user = ['None', 'SangIk', 'Dahoon']      # 사용자 수만큼 추가 id=1 -> names[1]

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

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

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # confidence : 정확도
        if (confidence < 50):
            if (id < 3):    # id : 시각장애인 - 1001~, 엄마 - 2001~ 이런식
                id = User[1]
            else:
                id = User[2]
            confidence = "  {0}%".format(round(100 - confidence))
        elif (confidence >= 50):
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n 프로그램 종료 및 정리")
cam.release()
cv2.destroyAllWindows()
