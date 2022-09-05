"""Utility Package for the project speed-detection

Usage:
	=================================================
		$ python3 utility.py videoplayback.mp4
	=================================================
"""
import cv2
import sys
import numpy as np

def draw_box(frame :np.ndarray, box :list, box_color :tuple):
	"""Draws box over the "frame" with box dimentions.

	frame is opencv frames (numpy).

	box is clockwise starting from left top x_cord , y_cord together. (8 members in list)
	"""
	cv2.line(frame, box[:2], box[2:4], box_color, 2)
	cv2.line(frame, box[4:6], box[6:8], box_color, 2)
	cv2.line(frame, box[:2], box[6:8], box_color, 2)
	cv2.line(frame, box[4:6], box[2:4], box_color, 2)

def infobox_over_object(frame :np.ndarray, box :list, text :str, box_color :tuple= (255, 100, 40), text_color :tuple=(255, 255, 255)):
	"""
	Draws box over and object in the "frame" and puts text above the drawn box over the object.

	frame is opencv frames (numpy).
	
	box is clockwise starting from left top x_cord , y_cord together. (8 members in list)
	"""
	cv2.rectangle(frame, (box[0]-1, box[1]), (box[0]+box[2]+1, box[1]+box[3]), box_color, 2)
	cv2.rectangle(frame, (box[0]-2, box[1] - 18), (box[0]+box[2]+2, box[1]), box_color, -1)
	cv2.putText(frame, text, (box[0], box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

def bb_intersection_over_union(boxA :np.ndarray, boxB :np.ndarray):
	"""Intersection over union caculation.
	"""
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def manhattan_distance(x1, y1, x2, y2):
	"""Calculating manhattan distance."""
	return abs(x2-x1)+abs(y2-y1)

def relative_line_position(x1 :int, y1 :int, points_2 :np.ndarray) -> float:
	m = (points_2[3]-points_2[1])/(points_2[2]-points_2[0])
	val = (y1-points_2[1] - m*(x1-points_2[0]))
	return val

def within_box(x1 :int, y1 :int, box) -> bool:
	"""Returns True if given point is within the box else returns False.
	"""
	if (x1 >= box[0] and x1 <= box[4] and y1 >= box[1] and y1 <= box[5] ):
		return True
	else:
		return False



if __name__ == "__main__":
	"""Use this main funciton to calculate bounding boxes.
	"""
	box1 = (583, 407, 556, 506, 185, 463, 313, 396)
	bounding_box = []
	print("Click in Frame to get it as list: ")

	def mouse_event(event, x, y, flag, params):
		if event == cv2.EVENT_LBUTTONDOWN:
			bounding_box.append(x)
			bounding_box.append(y)
			print(relative_line_position(x, y, box1[6:8]+box1[:2]))

	cv2.namedWindow('Frame')
	cv2.setMouseCallback('Frame', mouse_event)		

	cam = cv2.VideoCapture(sys.argv[1])
	while True:
		ret, frame = cam.read()
		if ret is False:
			break
		
		draw_box(frame, box1, (0, 255, 0))
		cv2.imshow('Frame', frame)
		key = cv2.waitKey(30)
		if key == ord('q'):
			break

	cv2.destroyAllWindows()
	cam.release()

	print('Bounding Box', bounding_box)