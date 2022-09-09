import numpy as np
import cv2

boxes = np.array([
    [[646,274],[732,268],[864,337],[634,348]], # right top
    [[634, 351], [868, 340], [934, 373], [641, 395]],
    [[641, 402], [939, 377], [1031, 426], [657, 464]],
    [[659, 469], [1031, 427], [1160, 493], [673, 552]],
    [[673, 559], [1159, 493], [1278, 563], [724, 715]],

    [[135, 443], [568, 515], [518, 708],   [4, 503]], # bottom left
    [[281, 380], [593, 434], [565, 514], [134, 442]],
    [[405, 327], [610, 338], [591, 437], [278, 379]],
    [[518, 280], [626, 288], [608, 338], [404, 328]]
]).reshape(9,-1, 1, 2)

FPS = 25

distances = np.array([
    28, # top most in left plane
    25,
    18,
    15,
    12,

    6, # bottom most in right plane
    5,
    7,
    13
])

def estimate_zone(prediction_judgement_arr :np.ndarray):
    """ 
    Returns [start, middle, middle_count, end]
    """
    start = -1
    middle = -1
    middle_count = -1
    end = -1
    for i in prediction_judgement_arr:
        if i[0] != -1:
            if (start == -1):
                start = i[0]
            if middle == -1 and start != i:
                middle = i[0]
            if middle == i[0]:
                middle_count += 1
            if end == -1 and middle != i and start != i:
                end = i[0]
    
    if end == -1:
        return -1, -1, -1, -1
    else:
        return start, middle, middle_count, end

def estimate_speed(curr_x :int, curr_y :int, future_prediction :np.ndarray) -> float:
    if future_prediction.shape[0] <= 4:
        return -1

    # finding current box the car is staying at
    current_box = -1
    for i, box in enumerate(boxes):
        if cv2.pointPolygonTest(box, (curr_x, curr_y), False) == 1:
            current_box = i

    # For each future predictions, the box area where the point is pointed
    prediction_judgement_arr = np.full((future_prediction.shape[0],1), -1)
    for i, f in enumerate(future_prediction):
        for j, box in enumerate(boxes):
            if cv2.pointPolygonTest(box, (f[0], f[1]), False) == 1:
                prediction_judgement_arr[i] = j
                break
    
    start, middle, middle_count, end = estimate_zone(prediction_judgement_arr)
    if end == -1:
        return -1
    
    return (distances[middle]/(middle_count/FPS))*3.6

box1 = np.array([[647, 349], [903, 343], [1005, 385], [650, 392]]).reshape(-1, 1, 2)
print(cv2.pointPolygonTest(box1, (777, 366), False))
print(cv2.pointPolygonTest(box1, (0, 0), False))

# box1 = np.array([[650, 391], [1003, 388], [1121, 452], [662, 478]]).reshape(-1, 1, 2)
# print(cv2.pointPolygonTest(box1, (845, 463), False))
# print(cv2.pointPolygonTest(box1, (0, 0), False))