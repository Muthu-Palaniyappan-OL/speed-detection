"""
Main Package for speed-detection project
"""
import cv2
from utility import draw_box, infobox_over_object
from objects import clean_object, get_id_speed, vehicle_centre
from speed_detection import check_speed
import numpy as np

"""Main Parameters for the project.
"""
BOX1 = np.array([583, 407, 556, 506, 185, 463, 313, 396])
BOX2 = np.array([675, 415, 963, 397, 1111, 465, 698, 512])
# BOX1 = np.array([488, 298, 612, 297, 512, 620, 14, 540])
# BOX2 = np.array([665, 303, 793, 293, 1275, 513, 746, 564])
FPS = 25
CONVERSION = 10
FRAMES_NUMBER = 1

if __name__ == "__main__":
	net = cv2.dnn_DetectionModel('yolov7-tiny.weights', 'yolov7-tiny.cfg')
	net.setInputSize((640, 640)) # resizing input which is sent into model into 640 X 640
	net.setInputScale(1/255.0) # RGB in range between 0 to 1
	net.setInputSwapRB(True) # swapping RGB for OpenCV

	class_list = [] # list of classes (cars, trucks) in coco dataset
	with open('coco.names') as f:
		class_list = f.read().split('\n')

	cam = cv2.VideoCapture('videoplayback.mp4') # video dataset

	while True:
		available, frame = cam.read()
		if not available:
			break

		draw_box(frame, BOX1, (20, 150, 200)) # Drawing Left Lane's Box
		draw_box(frame, BOX2, (20, 150, 200)) # Drawing Right Lane's Box

		# confidence threshhold 50%, Non Maximum Supression is 60%
		classes, confidences, boxes = net.detect(frame, confThreshold=0.5 ,nmsThreshold=0.6) 

		if len(classes) > 0: # if there is atleast 1 detection in the frame
			for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
				if class_list[classId] in ['car', 'truck']:
					centre = int(box[0] + box[2]/2), int(box[1] + box[3]/2)
					i, s = get_id_speed(*centre, 1 - confidence)

					# if i == 9:
					# 	print(i, (relative_line_position(*centre, BOX2[:4])))

					# vehicle_centre[i].speed = 6/(frame_difference_change(i, vehicle_centre, BOX2)*1/25) * 3.6
					if i ==9:
						print(i, s)
					
					if vehicle_centre[i].speed != 0:
						infobox_over_object(frame, box, '{} [{:.2f}]'.format(i, s ))
					else:
						infobox_over_object(frame, box, '{}'.format(i))
		
		clean_object()
		check_speed()
		

		cv2.imshow('Frame', frame)
		
		key = cv2.waitKey(0)
		if key == ord('q'):
			break


	cv2.destroyAllWindows()
	cam.release()
