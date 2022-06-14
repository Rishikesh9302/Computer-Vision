import cv2
import mediapipe as mp
import numpy as np
import time
import HandTrackingModule as htm
import autopy

smooth=7
frameR=100
wCam,hCam=640,480
wS,hS=autopy.screen.size()
# print(wS,hS)

plocX,plocY=0,0
clocX,clocY=0,0

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
detector=htm.handDetector(maxHands=1)
tipIds=[4,8,12,16,20]

while True:
    # 1. Get Hand Landmarks
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)

    # 2. Get Tip Of Index & Middle Fingers
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        #print(x1,y1,x2,y2)

        # 3. Check which fingers are up
        fingers = []
        # For Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # For 4 other fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        # 4. Only Index Finger Up -> Pointing/Moving Mode
        if fingers[1]==1 and fingers[2]==0:

        # 5. Convert Coordiates
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wS))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hS))
            # 6. Smoothening The Values
            clocX=plocX+(x3-plocX)/smooth
            clocY=plocY+(y3-plocY)/smooth

            # 7. Move Mouse

            autopy.mouse.move(wS-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY=clocX,clocY

        # 8. Both Index & Middle Fingers Up -> Selection/Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Calculate distance between fingers
            length,img,lineInfo=detector.findDistance(8,12,img)
            print(length)
            # 10. Click mouse if distance is small
            if length<40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

    # 11. Display the Frame Rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    # 12. Display the output
    cv2.imshow("Image",img)
    cv2.waitKey(1)