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

if __name__ == "__main__":
    box1 = (488, 298, 612, 297, 512, 620, 14, 540)
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
        
        draw_box_border(frame, box1)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.release()

    print('Bounding Box', bounding_box)