import cv2
import mediapipe as mp
import time
import PoseModule as pm

cap=cv2.VideoCapture('PoseVideos/2.mp4')
pTime=0
detector=pm.poseDetector()
while True:
    success,img=cap.read()
    img=detector.findPos(img)
    lmList=detector.findPosition(img)
    print(lmList)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(10)