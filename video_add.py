import cv2
import numpy as np
img = cv2.imread('nature.jpg')
img = cv2.resize(img,(640,360))
img_copy = np.copy(img)
cap = cv2.VideoCapture('tiger(2).mp4')

while(1):

    _,frame = cap.read()
    frame1 = frame
    frame = cv2.resize(frame,(640,360))
    frame1 = cv2.resize(frame1,(640,360))

    l_b = np.array([0,134,0])
    u_b = np.array([156, 255, 144])

    mask = cv2.inRange(frame,l_b,u_b)

    frame[mask!=0]=0
    
    a = img_copy-img
    img[mask==0]=0
    result = img+frame
    img = img+a
    
    cv2.imshow('frame',frame)
    cv2.imshow('Result', result)
    cv2.imshow('frame1',frame1)
    
    if cv2.waitKey(20)&0xFF==ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
