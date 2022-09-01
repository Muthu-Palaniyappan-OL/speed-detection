"""
Utility Package for the project speed-detection

Usage:
    =================================================
        $ python3 utility.py videoplayback.mp4
    =================================================
"""
import cv2
import sys

def draw_box_border(frame, box, color):
    cv2.line(frame, box[:2], box[2:4], color, 2)
    cv2.line(frame, box[4:6], box[6:8], color, 2)
    cv2.line(frame, box[:2], box[6:8], color, 2)
    cv2.line(frame, box[4:6], box[2:4], color, 2)

def within_box(x1 :int, y1 :int, box) -> bool:
    if (x1 >= box[0] and x1 <= box[4] and y1 >= box[1] and y1 <= box[5] ):
        return True
    else:
        return False

if __name__ == "__main__":
    box1 = (665, 303, 793, 293, 1275, 513, 746, 564)
    bounding_box = []
    print("Click in Frame to get it as list: ")

    def mouse_event(event, x, y, flag, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            bounding_box.append(x)
            bounding_box.append(y)

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_event)        

    cam = cv2.VideoCapture(sys.argv[1])
    while True:
        ret, frame = cam.read()
        if ret is False:
            break
        
        draw_box_border(frame, box1, (0, 255, 0))
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.release()

    print('Bounding Box', bounding_box)