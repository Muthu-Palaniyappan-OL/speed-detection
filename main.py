import cv2
net = cv2.dnn_DetectionModel('yolov7-tiny.weights', 'yolov7-tiny.cfg')
net.setInputSize((640, 640))
net.setInputScale(1/255.0)
net.setInputSwapRB(True)

# https://www.youtube.com/watch?v=wqctLW0Hb_0\n

names = []
with open('coco.names') as f:
    names = f.read().split('\n')

print('Classess Pretrained Neural Network Supports: ', names)
cam = cv2.VideoCapture('videoplayback.mp4')

while True:
    available, frame = cam.read()
    if not available:
        break

    frame = frame[:, 900:, :]
    
    classes, confidences, boxes = net.detect(frame, confThreshold=0.4 ,nmsThreshold=0.4)
        
    if(not len(classes) == 0):
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=1)
            cv2.putText(frame, names[classId]+' '+str(confidence), box[:2], cv2.FONT_HERSHEY_COMPLEX , 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (box[0] + int(box[2]/2), box[1]+ int(box[3]/2)), 3, (0, 0, 255), -1)
            center = (box[0] + int(box[2]/2), box[1]+ int(box[3]/2))
            print(center)
                
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    if key == 30:
        continue

cv2.destroyAllWindows()
cam.release()