import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self,minDetectionCon=0.5):


        self.minDetectionCon=minDetectionCon

        self.mpFaceDetection=mp.solutions.face_detection
        self.faceDetection=self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpDraw=mp.solutions.drawing_utils


    def findFaces(self,img,draw=True):

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        #print(self.results)
        bboxs=[]
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC=detection.location_data.relative_bounding_box
                h,w,c=img.shape
                bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
                bboxs.append([id,bbox,detection.score])
                img=self.fancy(img,bbox)
                cv2.putText(img,f'Confidence: {int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        return img,bboxs


    def fancy(selfself,img,bbox,l=20,t=5,rt=1):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        cv2.rectangle(img,bbox,(255,0,255),rt)
        #Top Left (x,y)
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t)
        #Top Right (x1,y)
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1,y + l), (255, 0, 255), t)
        #Bottom Left(x,y1)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        #Bottom Right(x1,y1)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 -l), (255, 0, 255), t)

        return img


def main():
    cap=cv2.VideoCapture("Videos/2.mp4")
    pTime = 0
    detector=FaceDetector()
    while True:
        success, img = cap.read()
        img,bboxs=detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(20)


if __name__=="__main__":
    main()