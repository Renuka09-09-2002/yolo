import streamlit as st
import cv2

import numpy as np

import os
def post_process(frame,outs,img,classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    boxes=[]
    confidences=[]
    classIDs=[]
    for out in outs: # calling each object boxes
        for detection in out: # calling each box
            score=detection[5:] # probability of 80 classes
            class_id=np.argmax(score) # max probability id
            confidence=score[class_id] # getting the confidance
            if confidence>0.7:         # if confidance >70% consider as that is valid bounding box
                centerX = int(detection[0] * frameWidth)  # before we pass the object we divided with frame width
                # these are the normalized values so multiply again
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )
                left = int(centerX - width/2)
                top = int(centerY - height/2)
                classIDs.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font=cv2.FONT_HERSHEY_SIMPLEX
    color=(255,0,0) # RGB
    for i in indexes:
        x,y,w,h=boxes[i]
        label=str(classes[classIDs[i]])
        confi=str(round(confidences[i],2))
        cv2.rectangle(img,(x,y),(x+w,y+h),color,5,i)   # (x,y): left,top     (x+w,y+h): right,bottom
        cv2.putText(img,label +" "+confi,(x,y),font,2,(255,255,255),3)
   



def yolo_out(modelConf, modelWeights, classesFile, image):
    net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    inpWidth = 416
    inpHeight = 416

    frame = cv2.imread(image)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    yolo_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(yolo_layers)
    post_process(frame,outs,img,classes)
    return frame, outs, img, classes

def display_detected_objects(frame, outs, img, classes):
    st.image(img, channels="RGB")  

def main():
    st.title("YOLO Object Detection App")

    modelConf = "yolov3-tiny.cfg"
    modelWeights = "yolov3-tiny.weights"
    classesFile = "coco.names"
    image = './images/horses.jpg'

    if st.button("Detect Objects"):
        frame, outs, img, classes = yolo_out(modelConf, modelWeights, classesFile, image)
        display_detected_objects(frame, outs, img, classes)

if __name__ == '__main__':
    main()
