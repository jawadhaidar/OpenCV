#import time cv mediapipe
import cv2 as cv
import mediapipe as mp
import time 





#use the webcam
cap=cv.VideoCapture(0)
#create object
mphands=mp.solutions.hands
#static true means both tracking and detection, false only detection
hands=mphands.Hands()
#draw
mpdraw=mp.solutions.drawing_utils

#frames
#previous time declaration
pt=0
#current time declaration
ct=0



while True:
    success,img=cap.read()
    #convert to RGB
    imgrgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    #use process method from hands object
    results=hands.process(imgrgb)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, pt in enumerate(handlms.landmark):
               #print(pt.x)
               h,w,c=img.shape
               cx,cy=int(pt.x*w),int(pt.y*h)
               #print(cx)
               if id==0:
                   cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
            

            #draw points and you can add connections 
            mpdraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)


    #get ct
    ct=time.time()
    #calculate fps
    fps=1/(ct-pt)
    #rest pt
    pt=ct

    #put text of the fps change to strng and to int
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    
    cv.imshow("Image",img)
    cv.waitKey(1)
    #if cv.waitKey(1) & 0xFF == ord('q'):
       # break
