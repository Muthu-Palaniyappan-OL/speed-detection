import numpy as np
import cv2

box1 = np.array([[647, 349], [903, 343], [1005, 385], [650, 392]]).reshape(-1, 1, 2)
print(cv2.pointPolygonTest(box1, (777, 366), False))
print(cv2.pointPolygonTest(box1, (0, 0), False))

box1 = np.array([[650, 391], [1003, 388], [1121, 452], [662, 478]]).reshape(-1, 1, 2)
print(cv2.pointPolygonTest(box1, (845, 463), False))
print(cv2.pointPolygonTest(box1, (0, 0), False))