import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import smtplib
from playsound import playsound
# import time
#import multiprocessing

sender_email = "min.minor413@gmail.com"
receiver_email = "leenaghatiya20347@acropolis.in"
password = "eezqniwpswktzjpu"
message="wake up sid ! Go look at your crops owner !! "
flag=True
s=False

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(sender_email, password)

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture(0)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
   

while True:    
    ret,frame = cap.read()   
    count += 1
    if count % 3 != 0:
        continue

    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
    print(px)
#c=class_list[d]

    for index,row in px.iterrows():
#        print(row)
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if 'dog' in c or 'cat' in c or 'bird' in c or 'horse' in c or 'sheep' in c or 'cow' in c or 'elephant' in c or 'bear' in c or 'zebra' in c or 'giraffe' in c:
            s=True    
            if flag :
                server.sendmail(sender_email, receiver_email,message)
                server.quit()
                flag=False
                
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) 
            cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
    if s:
        playsound('sound1.mp3')
        s=False

    

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

    
cap.release()
cv2.destroyAllWindows()

