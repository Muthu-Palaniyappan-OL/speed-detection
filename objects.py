from sys import maxsize
from typing import Tuple
from kalman_filter import KalmanFilter
from math import sqrt, pow

from utility import within_box

######################
CIRCLE_DISTANCE = 30
ID_STARTING = 0
######################

vehicle_list = []
vehicle_centers = dict()

def calc_distance(x1 :int, y1 :int, x2 :int, y2 :int) -> float:
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

def calc_speed(a :float, b :float):
    return sqrt(pow(a, 2) + pow(b, 2))

def vehicle_id_search(x :int, y :int) -> int:
    """
    Returns -1 if not found.
    Returns ID if it exists.
    """

    id = -1
    min_distance = maxsize

    for i in vehicle_centers:
        distance = 0
        print(vehicle_centers[i][1],vehicle_centers[i][2])
        distance = calc_distance(vehicle_centers[i][1],vehicle_centers[i][2], x, y)
        
        if (distance < min_distance and distance < CIRCLE_DISTANCE):
            min_distance = distance
            id = i
    
    return id

def get_object_id_and_speed(x :int, y :int, error :float) -> Tuple[int, float]:
    id = vehicle_id_search(x, y)
    ret = (0, 0)
    if id < 0:
        global ID_STARTING
        ID_STARTING += 1
        KF = KalmanFilter()
        vehicle_centers[ID_STARTING] = [KF, x, y, 0]
        id = ID_STARTING
        vehicle_centers[id][0].update(x, y, error)
        vehicle_list.append(id)
        ret = (id, 0)
    else:
            vehicle_centers[id][1] = x
            vehicle_centers[id][2] = y
            vehicle_centers[id][0].update(x, y, error)
            vx, vy = vehicle_centers[id][0].predict()
            ret = (id, calc_speed(vx[0], vy[0]))

    return ret

def check_each_vehicle(box):
    for i in vehicle_list:
        if vehicle_centers[i][0].worthy_enough() is True:
            pre = vehicle_centers[i][0].predict()
            x, y = (pre[0][1], pre[1][1])
            print(i, x, y)
            if (within_box(x, y, box) is False):
                vehicle_list.remove(i)
                print(i, vehicle_centers)
                del vehicle_centers[i]

    

if __name__ == "__main__":
    print(get_object_id_and_speed(894, 466, 0.1))
    print(get_object_id_and_speed(894, 466, 0.1))
    print(get_object_id_and_speed(894, 466, 0.1))
    print(get_object_id_and_speed(902, 476, 0.1))
    print(get_object_id_and_speed(913, 484, 0.1))