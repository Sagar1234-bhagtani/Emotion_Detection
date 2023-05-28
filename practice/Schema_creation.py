import os

import cv2
from tracker import *

import mysql.connector


MyDB= mysql.connector.connect(
    host="localhost",
    user = "root",
    password= "root",
    database = "sagar",
    auth_plugin="mysql_native_password"
)

MyCursor =MyDB.cursor()

#MyCursor.execute("CREATE TABLE IF NOT EXISTS datab (id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,Photo LONGBLOB NOT NULL,ArrivalDate datetime,Name varchar(25),Emotion varchar(15));")

#MyCursor.execute("CREATE TABLE IF NOT EXISTS Images (id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,Photo LONGBLOB NOT NULL);")
MyCursor.execute("CREATE TABLE IF NOT EXISTS datab (id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,Name varchar(25),Emotion varchar(15),ArrivalDate datetime,Photo LONGBLOB NOT NULL);")
#MyCursor.execute("CREATE TABLE IF NOT EXISTS datab (id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,Name varchar(25),Emotion varchar(15),Photo LONGBLOB NOT NULL);")


# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tracker=EuclideanDistTracker()

count=0
# Load the input video
cap = cv2.VideoCapture(0)

# Loop through each frame in the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    collect = []
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        collect.append([x,y,w,h])
        roi = frame[y:y + h, x: x + w]

    boxes_ids=tracker.update(collect)
    print(boxes_ids)
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        cv2.putText(frame,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    print(collect)
    # Display the output frame
    cv2.imshow('roi',roi)
    cv2.imshow('Face Detection', frame)
    cv2.imwrite("puthere/frame.jpg" , roi)
    with open("puthere/frame.jpg", "rb") as File:
        BinaryData = File.read()
    #SQLStatement = "INSERT INTO Images (Photo) VALUES (%s)"
    SQLStatement = "INSERT INTO datab (Name,Emotion,ArrivalDate,Photo) VALUES (%s,%s,NOW(),%s)"
   # SQLStatement = "INSERT INTO datab (Name,Emotion,Photo) VALUES (%s,%s,%s)"

    hola=("sagar","Neutral",BinaryData, )
    MyCursor.execute(SQLStatement,hola)
    MyDB.commit()
    os.remove("puthere/frame.jpg")
    count+=1
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()




