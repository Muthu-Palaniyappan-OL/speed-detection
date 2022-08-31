"""
Main Package for speed-detection project
"""

""" IMPORTANT PARAMETERS """

BOX1 = (488, 298, 612, 297, 512, 620, 14, 540)
BOX2 = (665, 303, 793, 293, 1275, 513, 746, 564)
FPS = 25

""" ========== """

import cv2
from utility import draw_box_border

net = cv2.dnn_DetectionModel('yolov7-tiny.weights', 'yolov7-tiny.cfg')
net.setInputSize((640, 640))
net.setInputScale(1/255.0)
net.setInputSwapRB(True)

frame_count = 1

names = []
with open('coco.names') as f:
    names = f.read().split('\n')

cam = cv2.VideoCapture('videoplayback.mp4')


while True:
    available, frame = cam.read()
    if not available:
        break

    classes, confidences, boxes = net.detect(frame, confThreshold=0.4 ,nmsThreshold=0.4)
        
    if(not len(classes) == 0):
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            draw_box_border(frame, BOX1, (20, 150, 200))
            draw_box_border(frame, BOX2, (20, 150, 200))
            if names[classId] in ['car', 'truck']:
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, (box[0] + int(box[2]/2), box[1]+ int(box[3]/2)), 4, (0, 0, 255), -1)
                center = (box[0] + int(box[2]/2), box[1]+ int(box[3]/2))
    
    print('Frame Count: ', frame_count)
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    if key == 30:
        continue

    frame_count += 1

cv2.destroyAllWindows()
cam.release()