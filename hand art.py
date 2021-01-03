import cv2
import numpy as np 
import imutils
from collections import deque


cap = cv2.VideoCapture(0)
pts = []

while(1):
    _, frame = cap.read()

    frame1 = cv2.flip(frame,1)
 #   frame1 = np.hstack([frame,flip]) 

    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([150,50,50])
    upper_blue = np.array([170,255,255])
    

    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    mask = cv2.erode(mask,None,iterations = 2)
    mask = cv2.dilate(mask,None,iterations = 2)
    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    res = cv2.bitwise_and(frame1,frame1, mask = mask)
    center = None
    if len(cnts)>0:
        c = max(cnts,key=cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
        if(radius > 10):
            cv2.circle(frame1,(int(x),int(y)),int(radius),(0,255,0),3)
            cv2.circle(frame1,center,5,(255,0,0),-1)
    pts.append(center)
    for i in range(1, len(pts))       :
        if pts[i-1] is None or pts[i] is None:
            continue
        cv2.line(frame1,pts[i-1],pts[i],(0,0,255),2)
    cv2.imshow('frame',frame1)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()