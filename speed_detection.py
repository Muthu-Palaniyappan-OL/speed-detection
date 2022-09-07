import numpy as np
from objects import vehicle_centre, OBJ


def relative_line_position(x1: int, y1: int, points_2: np.ndarray) -> float:
    m = (points_2[3]-points_2[1])/(points_2[2]-points_2[0])
    val = (y1-points_2[1] - m*(x1-points_2[0]))
    return val


def frame_difference_change(id: int, vehicle_centre: dict[int, OBJ], box, len=50):
    fpos = vehicle_centre[id].forward_position(len)
    res = 0
    for i, p in enumerate(fpos):
        t = relative_line_position(
            *p, box[:4])*relative_line_position(*p, box[4:])
        if (t/abs(t) == -1.0):
            res += 1

    return res


def check_speed():
    BOX2 = np.array([675, 415, 963, 397, 1111, 465, 698, 512])
    BOX1 = np.array([583, 407, 556, 506, 185, 463, 313, 396])
    for i in vehicle_centre:
        if relative_line_position(vehicle_centre[i].old_input_x, vehicle_centre[i].old_input_y, [642, 258, 641, 707]) > 0:
            if (relative_line_position(vehicle_centre[i].old_input_x, vehicle_centre[i].old_input_y, BOX2[:4]) > 0) and vehicle_centre[i].speed == 0:
                if (frame_difference_change(i, vehicle_centre, BOX2)*1/25) == 0:
                    break
                vehicle_centre[i].speed = (
                    14/(frame_difference_change(i, vehicle_centre, BOX2)*1/25)) * 3.6
        else:
            if (relative_line_position(vehicle_centre[i].old_input_x, vehicle_centre[i].old_input_y, BOX1[4:]) > 0) and vehicle_centre[i].speed == 0:
                if (frame_difference_change(i, vehicle_centre, BOX1)*1/25) == 0:
                    break
                vehicle_centre[i].speed = (
                    14/(frame_difference_change(i, vehicle_centre, BOX1)*1/25)) * 3.6
