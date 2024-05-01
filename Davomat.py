import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from deepface import DeepFace

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]

        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('DavomatCsv.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")

encodeListKnown = findEncoding(images)
print('Encoding Complete!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    imgS =cv2.resize(frame, (0,0),None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    model_name = 'ArcFace'

    try:
        # result = DeepFace.extract_faces(frame)
        resp = DeepFace.analyze(img_path=frame,
                                actions=['emotion'])
        for j in range(len(resp)):
            print(resp[j]['region'])
            emotion = resp[j]['dominant_emotion']
            x, y, w, h = resp[j]['region']['x'], resp[j]['region']['y'], resp[j]['region']['w'], resp[j]['region'][
                'h']

            # cv2.rectangle(frame, (x, y), (x + w, h + y), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
    except:
        print('Yuz topilmadi')

    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            # cv2.putText(frame, emotion, (x1 - 15, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 250, 250), 1)
            if faceDis[matchIndex] < 0.4:
                cv2.putText(frame, name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                markAttendance(name)
    quit = cv2.waitKey(1)&0xFF
    if quit == ord('q'):
        break
        # cv2.imshow('Image', imgS)

    cv2.imshow("Image",frame)

cap.release()
cv2.destroyAllWindows()

# faceLoc = face_recognition.face_locations(imgJeff)[0]
# encodeJeff = face_recognition.face_encodings(imgJeff)[0]
# cv2.rectangle(imgJeff,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest = face_recognition.face_locations(imgTEst)[0]
# encodeJeffTest = face_recognition.face_encodings(imgTEst)[0]
# cv2.rectangle(imgTEst,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# results = face_recognition.compare_faces([encodeJeff],encodeJeffTest)
# faceDis = face_recognition.face_distance([encodeJeff],encodeJeffTest)
# print(faceDis)
# cv2.putText(imgTEst, f"{results} {round(faceDis[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)