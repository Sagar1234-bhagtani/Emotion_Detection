from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
import time
import random
import pyshine as ps
from win32api import GetSystemMetrics

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('model.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

width = GetSystemMetrics(0)
height = GetSystemMetrics(1) - 50
wid = 1350
hei = 750

array = [["Looking angry...has someone blocked you?!", "Why so angry,do you have to work on saturdays also?",
          "Looking angry, seems you didn't get weekend movie tickets", "You are looking like angry bird.",
          "Looking angry...has someone blocked you?!", "Why so angry,do you have to work on saturdays also?",
          "Looking angry,seems you didn't get weekend movie tickets", "You are looking like angry bird.",
          "Looking angry...has someone blocked you?!", "Why so angry,do you have to work on saturdays also?"],
         # 0--#Angry

         ["", "", "", "", "", "", "", "", "", ""],  # disgust
         ["oooh..Fearful?! seems like the task is not completed yet", "Afraid..Don't be..Darr ke agey jeet hai",
          "oooh..Fearful?! seems like the task is not completed yet", "Afraid..Don't be..Darr ke agey jeet hai",
          "oooh..Fearful?! seems like the task is not completed yet", "Afraid..Don't be..Darr ke agey jeet hai",
          "oooh..Fearful?! seems like the task is not completed yet", "Afraid..Don't be..Darr ke agey jeet hai",
          "oooh..Fearful?! seems like the task is not completed yet", "Afraid..Don't be..Darr ke agey jeet hai"],
         # 3--Fear

         ["Happy!Seems like your salary got credited", "Happy!...redbull has given you wings?",
          "looks like you brushed your teeth perfectly",
          "Ahhaa!...Seems you got Netlink Rewards!", "Voilla!..your happiness tells all tasks are complete!",
          "Looks like your leaves have been approved! :)", "Seems happy...got to work from home",
          "Your smile shows you won lottery!", "Your face tells you just annoyed your coworker!",
          "Your smiling looks like it was a good day!)"],  # 4==Happy

         ["", "", "", "", "", "", "", "", "", ""],  # 5==Neutral

         ["Looking sad..has your coworker got an increment?", "Don't be sad, weekend is coming soon",
          "Why so sad..?had a breakup or a fight?", "smile often,you worked hard to brush today",
          "Employee-I want HIKE...Boss-download it from playstore", "Looking sad..Having a meeting at 7pm !!",
          "Don't worry apna time aaega :)",
          "Don't be sad, better luck next time for your leave approval",
          "Don't be sad you are doing a great job!)",
          "looks like you took a day off & now have 100s of mails!"],  # 6==Sad

         ['Looking surprised,"have fun at work"', "Don't get surprised..Gym zone is coming soon!",
          'Looking surprised,"have fun at work"', "Don't get surprised..Gym zone is coming soon!",
          'Looking surprised,"have fun at work"', "Don't get surprised..Gym zone is coming soon!",
          'Looking surprised,"have fun at work"', "Don't get surprised..Gym zone is coming soon!",
          'Looking surprised,"have fun at work"',
          "Don't get surprised..Gym zone is coming soon!"]]  # 7==Surprise

bg = cv2.imread("backk.jpeg")
mg = cv2.imread("backk.jpeg")
mg = cv2.resize(mg, (width, height))
bg = cv2.resize(bg, (width, height))

# Variables Intialization
ss = 0
r1 = 0
kk = 0
zz = 0

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi = roi_gray.astype('float') / 255.0  # Scale
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

        preds = emotion_model.predict(roi)[0]  # Yields one hot encoded result for 7 classes
        label = class_labels[preds.argmax()]  # Find the label
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        oo = preds.argmax()

        if time.time() - ss > 4:
            print(kk)
            # print("sagar")
            kk = ""
            ss = time.time()
            bg = cv2.resize(mg, (width, height))
            zz = 1  # free hai

        if (zz == 1 and oo != 4):  # if free
            r1 = random.randint(0, 9)
            zz = 0
            kk = array[oo][r1]
            ss = time.time()

        if (zz == 1 and oo == 4):
            kk = ""
            bg = cv2.resize(mg, (width, height))
        print(str(zz) + " " + str(oo))

    bg = ps.putBText(bg, str(kk), text_offset_x=int((width / 2) - 300), text_offset_y=int(height - 250), vspace=10,
                     hspace=10, font_scale=1.5,
                     background_RGB=(255, 255, 255), text_RGB=(222, 49, 99))
    frame = cv2.resize(frame, (wid, hei))
    cv2.imshow('.', frame)
    cv2.imshow('Netlink', bg)

    # cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()