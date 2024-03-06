import streamlit as st
import cv2
import numpy as np

def post_process(frame, outs, classes):
    frameHeight, frameWidth = frame.shape[:2]
    boxes = []
    confidences = []
    classIDs = []
    for out in outs:  # calling each object boxes
        for detection in out:  # calling each box
            scores = detection[5:]  # probability of 80 classes
            class_id = np.argmax(scores)  # max probability id
            confidence = scores[class_id]  # getting the confidence
            if confidence > 0.5:  # if confidence >50% consider as that is valid bounding box
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(centerX - width / 2)
                top = int(centerY - height / 2)
                classIDs.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)  # RGB
    for i in indexes:
        x, y, w, h = boxes[i[0]]
        label = str(classes[classIDs[i[0]]])
        confi = str(round(confidences[i[0]], 2))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label + " " + confi, (x, y - 10), font, 1, (255, 255, 255), 2)


def yolo_out(modelConf, modelWeights, video_path):
    net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
    

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        inpWidth = frame.shape[1]
        inpHeight = frame.shape[0]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        yolo_layers = net.getUnconnectedOutLayersNames()
        outs = net.forward(yolo_layers)
        post_process(frame, outs, classes)

        cv2.imshow('YOLO Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    st.title("YOLO Object Detection App for Video")

    modelConf = "yolov3.cfg"
    modelWeights = "yolov3.weights"
    
    video_path = './images/los_angeles (1).mp4'

    yolo_out(modelConf, modelWeights, video_path)


if __name__ == '__main__':
    main()
