from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
import random
import pyshine as ps
from win32api import GetSystemMetrics
import time

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_interpreter = tf.lite.Interpreter(model_path="ambala.tflite")
emotion_interpreter.allocate_tensors()

emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Test the model on input data.
emotion_input_shape = emotion_input_details[0]['shape']
ss = time.time()  # for Time

array = [["Looking angry...has someone blocked uhh!!", "Why so angry,do you h   ave to work on saturday also",
          "Looking angry,seems uhh didn't get weekend movie tickets", "You looking like angry bird.",
          "Looking angry...has someone blocked uhh!!", "Why so angry,do you have to work on saturday also",
          "Looking angry,seems uhh didn't get weekend movie tickets", "You looking like angry bird.",
          "Looking angry...has someone blocked uhh!!", "Why so angry,do you have to work on saturday also"],
         # 0--#Angry

         ["", "", "", "", "", "", "", "", "", ""],  # disgust
         ["Fearful!! Seems not completed the task", "Afraid..Don,t be..Darr k agey jeet h",
          "Fearful!! Seems not completed the task", "Afraid..Don't be..Darr k agey jeet h",
          "Fearful!! Seems not completed the task", "Afraid..Don't be..Darr k agey jeet h",
          "Fearful!! Seems not completed the task", "Afraid..Don't be..Darr k agey jeet h",
          "Fearful!! Seems not completed the task", "Afraid..Don't be..Darr k agey jeet h"],  # 3--Fear

         ["Happy!Seems your salary got credited", "Happy...had redbull gave you wings",
          "3. looks like you brushed teeth perfectly:)",
          "Ahhaa...Seems you got Netlink Rewards", "Voila..your happiness shows task got completed",
          "Looks like your vacation leaves got approved:)", "Seems happy...got work from home",
          "Your smile shows you won lottery", "Your face tells you just annoyed your coworker",
          "Your smile shows, today time passed with work meetings;)"],  # 4==Happy

         ["", "", "", "", "", "", "", "", "", ""],  # 5==Neutral

         ["Looking sad!Does your coworker got an increment", "Dont be sad,friday is coming soon",
          "Why so sad..had a breakup or a fight with spouse", "Show open smile,you worked hard to brush today",
          "Employee-I want HIKE...Boss-download it from playstore", "Looking sad..Having a meeting at 7pm !!",
          "Unhappy! Seems you got promotion on same salary",
          "Don't be sad, betetr luck next time for vacation leave approval",
          "Don't be sad you are working enough not to be homeless:)",
          "I think you took 1 day off & now has 100 messages"],  # 6==Sad

         ['Looking surprised,"have fun at work"', "Don't get surprised..soon compnany introducing gym zone",
          'Looking surprised,"have fun at work"', "Don't get surprised..soon compnany introducing gym zone",
          'Looking surprised,"have fun at work"', "Don't get surprised..soon compnany introducing gym zone",
          'Looking surprised,"have fun at work"', "Don't get surprised..soon compnany introducing gym zone",
          'Looking surprised,"have fun at work"',
          "Don't get surprised..soon compnany introducing gym zone"]]  # 7==Surprise

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

width =GetSystemMetrics(0)
height=GetSystemMetrics(1)
wid=1500
hei=320
# wid = 1600
# hei = 800
# width = 1920
# height = 1080

bg = cv2.imread("background.jpeg")
mg = cv2.imread("background.jpeg")
mg = cv2.resize(mg, (width, height))
print(width)
print(height)
r1 = 0
pp = 0
uu = 0
oo = 0
zz = 0
while True:
    ret, frame = cap.read()
    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # bg = cv2.imread("background.png")
    bg = cv2.resize(bg, (width, height))

    if (time.time() - ss) > 10:
        zz = 1
        cv2.imshow('Single Channel Window', mg)
        # bg=mg
        # cv2.imshow('Single Channel Window', bg)
        print("sagar")
        print(ss)
        ss = time.time()
        if oo == 4:
            cv2.imshow('Single Channel Window', mg)
    # if(zz==1):
    #     bg=mg
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi = roi_gray.astype('float') / 255.0  # Scale
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        # preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 7 classes
        emotion_label = class_labels[emotion_preds.argmax()]  # Find the label
        emotion_label_position = (x, y)
        cv2.putText(frame, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        oo = emotion_preds.argmax()
        if oo != pp:
            ss = time.time()
            r1 = random.randint(0, 9)

            # if(ss>8):
            #     cv2.imshow('Single Channel Window', mg)

        pp = oo
        kk = array[oo][r1]

        bg = ps.putBText(bg, kk, text_offset_x=int((width / 2) - 250), text_offset_y=int(height - 250), vspace=10,
                         hspace=10, font_scale=1.5,
                         background_RGB=(255, 255, 255), text_RGB=(222, 49, 99))
        # if(oo==4):
        #     cv2.imshow('Single Channel Window', mg)
    # if (time.time()-ss)>10:
    if (oo != 4):
        cv2.imshow('Single Channel Window', bg)
        # print("sagar")
        # print(ss)
        # ss=time.time()
        # cv2.waitKey(0)
    if (ss > 8):
        bg = mg
    frame = cv2.resize(frame, (wid, hei))
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
        break
cap.release()
cv2.destroyAllWindows()