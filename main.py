"""
Main Package for speed-detection project
"""
from re import X
import cv2
from objects import check_each_vehicle, get_object_id_and_speed, vehicle_list, vehicle_centers
from utility import draw_box_border, within_box
import numpy as np

################# Important Paramerter #####################

BOX1 = np.array([488, 298, 612, 297, 512, 620, 14, 540])
BOX2 = np.array([665, 303, 793, 293, 1275, 513, 746, 564])
FPS = 25
KM_PER_PIXEL = 25

############################################################

if __name__ == "__main__":
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

        draw_box_border(frame, BOX1, (20, 150, 200))
        draw_box_border(frame, BOX2, (20, 150, 200))

        classes, confidences, boxes = net.detect(frame, confThreshold=0.4 ,nmsThreshold=0.4)

        ids = []
            
        if(not len(classes) == 0):
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                if (within_box(box[0], box[1], BOX2) or within_box(box[0], box[1], BOX1)):
                    cv2.rectangle(frame, box, color=(0, 0, 0), thickness=2)
                    cv2.circle(frame, (box[0] + int(box[2]/2), box[1]+ int(box[3]/2)), 4, (0, 0, 255), -1)
                    center = (box[0] + int(box[2]/2), box[1]+ int(box[3]/2))
                    obj_id, speed = get_object_id_and_speed(center[0], center[1], 1 - confidence)
                    ids.append(obj_id)
                    vehicle_centers[obj_id][3] = 0
                    cv2.rectangle(frame, (box[0], box[1]-20, 80, 20), color=(0, 0, 0), thickness=-1)
                    cv2.putText(frame, str(obj_id) + ' [' + str(speed//KM_PER_PIXEL) + ']', (box[0]+2, box[1]-5), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 1)
            
            for i in vehicle_list:
                if i not in ids:
                    vehicle_centers[i][3] += 1
                    if vehicle_centers[i][0].worthy_enough() is True:
                        x, y = vehicle_centers[i][0].predict()
                        vehicle_centers[i][1] = x[0]
                        vehicle_centers[i][2] = y[0]
                    if vehicle_centers[i][3] >= 10:
                        del vehicle_centers[i]
                        vehicle_list.remove(i) 

        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(int(1/FPS * 1000))
        if key == ord('q'):
            break
        if key == 30:
            continue

        frame_count += 1

    cv2.destroyAllWindows()
    cam.release()