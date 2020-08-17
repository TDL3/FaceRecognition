#!/bin/env python3
import json
import os
import re

import cv2
import numpy
from PIL import Image


class FaceRecognition():
    
    def __init__(self):
        super().__init__()
        # fix for cap_msmf.cpp (435) `anonymous-namespace' warning (windows only).
        if os.name == "nt":
            self.video_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.video_stream = cv2.VideoCapture(0)
        self.face_detector = cv2.CascadeClassifier("./face_detector/haarcascade_frontalface_default.xml")
        self.model = "./trained_model/trained_model.yml"
        self.face_dict = {}
        self.dict_path = r"./face_dict.json"
        self.dataset_path = r"./dataset"
        self.trained_model_path = r'./trained_model/trained_model.yml'
        
    def menu(self):
        print('*' * 31)
        print('''
        1, Face collection
        2, Train model
        3, Face recognition
        d, Delete saved data
        q, Quit
        ''')
        print('*' * 31)
        
    def release_resources(self):
        self.video_stream.release()
        cv2.destroyAllWindows()
        
    def save_dict(self, dict):
        with open(self.dict_path, 'w') as fp:
            json.dump(dict, fp, sort_keys=True, indent=4)
            
    def load_dict(self):
        if os.stat(self.dict_path).st_size == 0:
            return {}
        with open(self.dict_path, 'r') as fp:
            return json.load(fp)
    
    def clear_dict(self):
        with open(self.dict_path, 'w') as fp:
            json.dump({}, fp)
    
    

    def collect_faces(self):
        while True:
            print("Enter 'q' to stop")
            face_id = input('Enter ID(integer) of the face: ')
            if face_id == 'q':
                break
            face_name = input('Enter name of the face: ')
            if face_name == 'q':
                break
            self.face_dict[face_id] = face_name
            count = 0
            while count < 60:
                _, image_frame = self.video_stream.read()
                gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    path = f"{self.dataset_path}/[{face_id}]{face_name} {str(count)}.png"
                    cv2.imwrite(path, gray[y:y + h, x:x + w])
                    cv2.imshow('Face Collection', image_frame)
                    count += 1
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            print(f'Face data collection for {self.face_dict[face_id]} finished')
            self.save_dict(self.face_dict)
            cv2.destroyAllWindows()
        self.release_resources()
        
    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = numpy.array(PIL_img, 'uint8')
                id = int(re.search(r"\[\d", imagePath)[0].replace("[", ""))
                faces = self.face_detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
            return faceSamples, ids

        faces, ids = getImagesAndLabels(self.dataset_path)
        recognizer.train(faces, numpy.array(ids))
        recognizer.save(self.trained_model_path)
        print("Model trained!")
        
    # 人脸识别
    def predict_face(self, mydict):

        print("Enter 'q' to quit")
        mydict = mydict
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.trained_model_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cam = cv2.VideoCapture(0)
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

                if conf < 80:
                    if str(Id) in mydict:
                        Id = mydict[str(Id)]
                else:
                    Id = "Unknown"
                cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
                cv2.putText(im, str(Id), (x, y - 40), font, 2, (255, 255, 255), 3)
                
            cv2.imshow('Face Recognition', im)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        self.release_resources()
        
    def del_file(self, path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path,i)
            if os.path.isdir(c_path):
                self.del_file(c_path)
            else:
                os.remove(c_path)
        self.clear_dict()
        print("Data deleted")

        
    def wrapper(self):
        self.face_dict = self.load_dict()
        while True:
            self.menu()
            num = input("Enter a choice: ")
            if num == '1':
                self.collect_faces()
            elif num == '2':
                self.train_model()
            elif num == '3':
                self.predict_face(self.face_dict)
            elif num == 'd':
                self.del_file(self.dataset_path)
            elif num == 'q':
               self.release_resources()
               break
            

if __name__ == "__main__":
    fr = FaceRecognition()
    fr.wrapper()
