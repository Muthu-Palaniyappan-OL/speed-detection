"""
Python File Version of Main.ipynb
"""

import cv2
import numpy as np
import sort # sort.py
import speed_estimation as se
neural_network = cv2.dnn_DetectionModel('yolov7-tiny.weights', 'yolov7-tiny.cfg') # yolov7 pretrained model
neural_network.setInputSize((640, 640)) # input size for yolov7 pretrained model
neural_network.setInputScale(1.0/255.0) # input range for yolov7 pretrained model
neural_network.setInputSwapRB(True) # opencv's BGR to RGB


def infobox_over_object(frame: np.ndarray, box: list, text: str, box_color: tuple = (255, 100, 40), text_color: tuple = (255, 255, 255)):
    """
    Draws box over and object in the "frame" and puts text above the drawn box over the object.

    frame is opencv frames (numpy).

    box is clockwise starting from left top x_cord , y_cord together. (8 members in list)
    """
    cv2.rectangle(frame, (box[0]-1, box[1]),
                  (box[0]+box[2]+1, box[1]+box[3]), box_color, 2)
    cv2.rectangle(frame, (box[0]-2, box[1] - 18),
                  (box[0]+box[2]+2, box[1]), box_color, -1)
    cv2.putText(frame, text, (box[0], box[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


def draw_boxes(frame :np.ndarray, box :list[int], box_color: tuple = (255, 100, 40)):
    # cv2.rectangle(frame, (box[0][0][0], box[0][0][1]), (box[2][0][0], box[2][0][1]), box_color, 2)
    # for i in range(0, len(initial_box) - 2, 2):
    #         cv2.line(frame, initial_box[i:i+2], initial_box[i+2:i+4], (0, 255, 0), 2)
    for i in range(0, len(box) -1, 1):
            cv2.line(frame, (box[i][0][0], box[i][0][1]), (box[i+1][0][0], box[i+1][0][1]), (0, 255, 0), 2)
    cv2.line(frame, (box[len(box) -1][0][0], box[len(box) -1][0][1]), (box[0][0][0], box[0][0][1]), (0, 255, 0), 2)


cam = cv2.VideoCapture('videoplayback.mp4')
s = sort.SORT(Tlost_max=30, iou_min=0.1)
video_cod = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter('captured_video.mp4', video_cod, int(cam.get(5) * 0.40),
                               (int(cam.get(3)), int(cam.get(4))))

while True:
    ret_status, frame = cam.read()
    if not ret_status:
        break

    # detect from Neural Network
    classes, confidences, boxes = neural_network.detect(frame, confThreshold=0.5, nmsThreshold=0.4)

    # drawing all the zone boxes
    for i in se.boxes:
        draw_boxes(frame, i)

    # if no detections break
    if len(classes) == 0:
        break

    # Converting top_left edge oriented boxes to center oriented
    boxes[:, 2] = boxes[:, 2] / np.full(boxes[:, 2].shape, 2.0)
    boxes[:, 3] = boxes[:, 3] / np.full(boxes[:, 3].shape, 2.0)
    boxes[:, 0] += boxes[:, 2]
    boxes[:, 1] += boxes[:, 3]

    # sending it into sort algorithm
    model_predictions = np.concatenate((boxes, confidences.reshape(-1, 1)), axis=1, dtype=np.float16, casting='unsafe')
    res = s.update(model_predictions)

    # converting center oriented output into topleft edge oriented
    res[:, 0] = res[:, 0] - res[:, 2]
    res[:, 1] = res[:, 1] - res[:, 3]
    res[:, 6] = res[:, 6] - res[:, 2]
    res[:, 7] = res[:, 7] - res[:, 3]
    res[:, 2] = res[:, 2] * np.full(res[:, 2].shape, 2.0)
    res[:, 3] = res[:, 3] * np.full(res[:, 3].shape, 2.0)

    for i in res:

        speed = se.estimate_speed(s.tracker[i[5]].old_x, s.tracker[i[5]].old_y, s.tracker[i[5]].future_predictions())
        if speed != -1:
            s.tracker[i[5]].speed = speed

        # Drawing Blue Boxes Which are acctual car detections from yolov7 model (if speed is set it writes else it doesn't)
        infobox_over_object(frame, (int(i[0]), int(i[1]), int(i[2]), int(i[3])),
                            "{:d} {}".format(int(i[5]), ("[{:.2f}]".format(i[9]) if i[9] != 0 else "")))
        # Drawing Red Boxes Which are detections predicted by kalaman filter (if speed is set it writes else it doesn't)
        infobox_over_object(frame, (int(i[6]), int(i[7]), int(i[2]), int(i[3])),
                            "{:d} {}".format(int(i[5]), ("[{:.2f}]".format(i[9]) if i[9] != 0 else "")),
                            box_color=(0, 0, 255))

    cv2.imshow('Frame', frame)
    video_output.write(frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cam.release()
video_output.release()
cv2.destroyAllWindows()