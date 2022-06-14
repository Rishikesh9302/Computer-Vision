import cv2
import mediapipe as mp
import os
import time
import HandTrackingModule as htm


#####
wCam,hCam=640,480
#####

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath="FingerPhotos"
myList=os.listdir(folderPath)
print(myList)
overlaylist=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlaylist.append(image)
print(len(overlaylist))

pTime=0
detector=htm.handDetector(detectionCon=0.75)

tipIds=[4,8,12,16,20]
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    #print(lmList)

    if len(lmList)!=0:
        fingers=[]

        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalF=fingers.count(1)
        print(totalF)


        h,w,c=overlaylist[totalF-1].shape

        img[0:h,0:w]=overlaylist[totalF-1]

        cv2.rectangle(img,(20,225),(170,425),(255,0,0),cv2.FILLED)
        cv2.putText(img,str(totalF),(45,375),cv2.FONT_HERSHEY_COMPLEX_SMALL,7,(0,255,0),20)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)