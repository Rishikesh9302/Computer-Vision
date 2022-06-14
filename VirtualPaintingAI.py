import cv2
import time
import numpy as np
import os
import HandTrackingModule as htm

# ----------------------
brushThickness=15
eraserThickness=50
# ----------------------


folderPath="HDR"
myList=os.listdir(folderPath)
#print(myList)
overlayList=[]

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))
header=overlayList[0]

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector=htm.handDetector(detectionCon=0.85)
drawColor=(255,0,255)

xp,yp=0,0
imgCanvas=np.zeros((720,1280,3),np.uint8)

while True:

    # 1. Importing The Images
    success,img=cap.read()
    img=cv2.flip(img,1)

    # 2. Finding Hand Landmarks
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)

    if len(lmList)!=0:
        #print(lmList)

        #Intialising Index & Middle Finger Tips
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]


        # 3. Check which fingers are up

        fingers=detector.fingersUp()
        #print(fingers)

        # 4. Tool Selection Mode -> 2 fingers (Index & Middle) are UP!
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("SELCTION MODE")
            #Checking for click
            if y1<125:
                if 250<x1<450:
                    header=overlayList[0]
                    drawColor=(255,0,255)
                if 550<x1<750:
                    header=overlayList[1]
                    drawColor=((255,0,0))
                if 800<x1<950:
                    header=overlayList[2]
                    drawColor=(0,255,0)
                if 1050<x1<1200:
                    header=overlayList[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. Drawing Mode -> 1 finger (Index ONLY) is UP!
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("DRAWING MODE")
            if xp == 0 and yp == 0:
                xp , yp= x1 , y1

            if drawColor==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp= x1, y1

    imgGy=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    a,imgInverse=cv2.threshold(imgGy,50,255,cv2.THRESH_BINARY_INV)
    imgInverse=cv2.cvtColor(imgInverse,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInverse)
    img=cv2.bitwise_or(img,imgCanvas)


    #Initialising the header images
    img[0:125,0:1280]=header
    #img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.waitKey(1)