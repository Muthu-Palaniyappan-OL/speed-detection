import enum
from sys import maxsize
from typing import Tuple
from kalman_filter import KalmanFilter
import numpy as np

from utility import manhattan_distance, relative_line_position

class OBJ(KalmanFilter):
	INITIAL_OBJECT_ID = 1
	def __init__(self, dt=1, initial_error=1000, old_input_x=0, old_input_y=0):
		super().__init__(dt, initial_error, old_input_x, old_input_y)
		self.id = OBJ.INITIAL_OBJECT_ID
		self.time_passed_after = 0
		self.assigned = False
		OBJ.INITIAL_OBJECT_ID += 1
		
vehicle_centre : dict[int, OBJ] = dict()
DISTANCE_RADIOUS = 100
Max_TLOST_FRAMES = 19

def get_id_speed(x, y, error):
	min = maxsize
	id = -1
	for i in vehicle_centre:
		dist = manhattan_distance(x, y, *vehicle_centre[i].predict_distance())
		if dist < min and dist < DISTANCE_RADIOUS and vehicle_centre[i].assigned == False:
			id = i
			min = dist
		
	if id == -1:
		K = OBJ(old_input_x=x, old_input_y=y)
		K.update(x, y, error)
		vehicle_centre[K.id] = K
		K.assigned = True
		K.time_passed_after = 0
		return K.id, K.predict_velocity()
	else:
		vehicle_centre[id].assigned = True
		vehicle_centre[id].time_passed_after = 0
		vehicle_centre[id].update(x, y, error)
		return vehicle_centre[id].id, vehicle_centre[i].predict_velocity()
	
def clean_object():
	delete_ids = []
	for i in vehicle_centre:
		if vehicle_centre[i].assigned == False:
			vehicle_centre[i].update(*vehicle_centre[i].predict_distance(), 0.5)
			vehicle_centre[i].time_passed_after += 1
		if vehicle_centre[i].time_passed_after > Max_TLOST_FRAMES:
			delete_ids.append(i)
		vehicle_centre[i].assigned = False
	
	for i in delete_ids:
		del vehicle_centre[i]

def frame_difference_change(id :int, vehicle_centre :dict[int, OBJ], box, len = 10):
	fpos = vehicle_centre[id].forward_position(len)
	res = 0
	for i, p in enumerate(fpos):
		t = relative_line_position(*p, box[:4])*relative_line_position(*p, box[4:])
		if (t/abs(t) == -1.0):
			res += 1
	
	return res

if __name__ == "__main__":
	BOX2 = np.array([675, 415, 963, 397, 1111, 465, 698, 512])
	get_id_speed(692, 315, 0.1)
	clean_object()
	get_id_speed(692, 315, 0.1)
	clean_object()
	get_id_speed(691, 316, 0.1)
	clean_object()
	get_id_speed(694, 319, 0.1)
	clean_object()
	get_id_speed(701, 334, 0.1)
	clean_object()
	get_id_speed(706, 341, 0.1)
	clean_object()
	get_id_speed(708, 344, 0.1)
	clean_object()
	get_id_speed(708, 345, 0.1)
	clean_object()
	get_id_speed(709, 348, 0.1)
	clean_object()
	get_id_speed(710, 350, 0.1)
	clean_object()
	get_id_speed(712, 353, 0.1)
	clean_object()
	get_id_speed(713, 354, 0.1)
	clean_object()
	get_id_speed(716, 357, 0.1)
	clean_object()
	get_id_speed(692, 315, 0.1)
	clean_object()
	get_id_speed(692, 315, 0.1)
	clean_object()
	get_id_speed(692, 315, 0.1)
	clean_object()
	get_id_speed(691, 316, 0.1)
	clean_object()
	get_id_speed(694, 319, 0.1)
	clean_object()
	get_id_speed(701, 334, 0.1)
	clean_object()
	get_id_speed(706, 341, 0.1)
	clean_object()
	get_id_speed(708, 344, 0.1)
	clean_object()
	get_id_speed(708, 345, 0.1)
	clean_object()
	get_id_speed(709, 348, 0.1)
	clean_object()
	get_id_speed(710, 350, 0.1)
	clean_object()
	get_id_speed(712, 353, 0.1)
	clean_object()
	get_id_speed(713, 354, 0.1)
	clean_object()
	get_id_speed(716, 357, 0.1)
	clean_object()
	print(frame_difference_change(1, vehicle_centre, BOX2) * 1/25 * 3.6 * 6)
