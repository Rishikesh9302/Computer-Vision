import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture("Videos/1.mp4")

pTime=0

mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection()
mpDraw=mp.solutions.drawing_utils


while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #print(id,detection)
            print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            #mpDraw.draw_detection(img,detection)
            bboxC=detection.location_data.relative_bounding_box
            h,w,c=img.shape
            bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f'Confidence: {int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(10)