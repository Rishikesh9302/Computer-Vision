import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, mode=False,max_num_faces=2,refine_land=False,detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.max_num_faces = max_num_faces
        self.refine_land = refine_land
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(self.mode, self.max_num_faces, self.refine_land,  self.detectionCon, self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def findFaceMesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face=[]
                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img,faces



def main():
    cap = cv2.VideoCapture('Videos/2.mp4')
    pTime = 0
    detector=FaceMeshDetector(max_num_faces=2)
    while True:
        success,img = cap.read()
        img,faces=detector.findFaceMesh(img)
        if len(faces)!=0:
            print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(5)

if __name__=="__main__":
    main()