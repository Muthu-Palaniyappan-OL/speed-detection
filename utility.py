"""
Utility Package for the project speed-detection

Usage:
    =================================================
        >>> $ python3 utility.py videoplayback.mp4
    =================================================
"""
import cv2
import sys
import numpy as np

if __name__ == "__main__":
    initial_box = np.array([[405, 327], [610, 338], [591, 437], [278, 379]], dtype=int).reshape(1, 8)[0]
    # initial_box = np.array([647, 349, 903, 343, 1005, 385, 650, 392], dtype=int)
    bounding_box = np.array([], dtype=int)
    print("Click in Frame to get it as list: ")

    def mouse_event(event, x, y, flag, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            global bounding_box
            bounding_box = np.concatenate((bounding_box, np.array([x, y])), axis=0)

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_event)        

    cam = cv2.VideoCapture(sys.argv[1])

    while True:
        ret, frame = cam.read()
        if ret is False:
            break
        
        for i in range(0, len(initial_box) - 2, 2):
            cv2.line(frame, initial_box[i:i+2], initial_box[i+2:i+4], (0, 255, 0), 2)
        
        if len(bounding_box) >= 4:
            for i in range(0, len(bounding_box) - 2, 2):
                cv2.line(frame, bounding_box[i:i+2], bounding_box[i+2:i+4], (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.release()

    print('Bounding Box', bounding_box[:8])