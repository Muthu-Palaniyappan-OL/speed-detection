"""
# Simple Online Realtime Tracking

Usage:
    s = sort.SORT()
    s.update(model_predictions)
"""
import numpy as np
from typing import Tuple


class Kalaman_Filter:
    """ kalman Filter

    ### 2D Constant linear velocity model based kalman filter

    Usage:

        kf = Kalaman_Filter(old_x=1, old_y=1)
        kf.update(3, 3, 0.1)
        kf.update(4, 4, 0.3)
        kf.update(7, 7, 0.1)
        kf.predict_position()
        kf.predict_speed()

    """

    # Static Variable
    dt: int = 1
    A: np.ndarray = np.array([[1, dt], [0, 1]]) # Constant Linear Velocity Model

    # Constructor
    def __init__(self, dt=1, initial_error=1000, old_x=0, old_y=0) -> None:
        self.dt = dt
        self.old_x = old_x
        self.old_y = old_y
        self.Xx = np.array([old_x, 0], dtype=np.float16) # State Variable for x_cord
        self.Xy = np.array([old_y, 0], dtype=np.float16) # State Variable for y_cord
        self.Px = np.eye(2, dtype=np.float16)*initial_error # Initial covariance matrix of x_cord
        self.Py = np.eye(2, dtype=np.float16)*initial_error # Initial covariance matrix of y_cord

    def __priori_estimate(self, X: np.ndarray) -> np.ndarray:
        return self.A.dot(X.transpose())

    def priori_estimate(self, X: np.ndarray) -> np.ndarray:
        return self.A.dot(X.transpose())

    def __estimate_covariance(self, P: np.ndarray) -> np.ndarray:
        return self.A.dot(P.dot(self.A.transpose()))

    def __calculate_kalman_gain(self, P: np.ndarray, R: np.ndarray) -> np.ndarray:
        return np.divide(P, (P+R), where=((P+R) != 0))

    def __postirior_estimate(self, Y: np.ndarray, K: np.ndarray, X: np.ndarray) -> np.ndarray:
        return (X + K.dot(Y.transpose() - X.transpose()).transpose())

    def __corrected_covariance(self, P: np.ndarray, K: np.ndarray) -> np.ndarray:
        return ((np.eye(2) - K).dot(P))

    def predict_position(self) -> Tuple[int, int]:
        """ 
        Returns predicted position w.r.t to both x_cord, y_cord
        """
        return round(self.__priori_estimate(self.Xx)[0]), round(self.__priori_estimate(self.Xy)[0])

    def predict_speed(self) -> Tuple[int, int]:
        """ 
        Returns predicted speed w.r.t to both x_cord, y_cord
        """
        return self.__priori_estimate(self.Xx)[1], self.__priori_estimate(self.Xy)[1]

    def __diagonal_filter(self, K: np.ndarray) -> np.ndarray:
        """ 
        # Extracts only the diagonal elements of the matrix.

        Input = [[3, 3]
                [3, 3]]
        
        Ouput = [[3, 0]
                [0, 3]]
        """
        top_left = np.array([[1, 0], [0, 0]])
        bottom_right = np.array([[0, 0], [0, 1]])
        return top_left.dot(K.dot(top_left)) + bottom_right.dot(K.dot(bottom_right))

    def update(self, x1: int, y1: int, error: float) -> None:
        """ 
        # Update the kalman filter

        More updates better the kalman filter predicts.

        
        Input: (X_cord, Y_cord, error)
        
        Error value is betweeen 0 and 1

        Output: None

        For more information: https://arxiv.org/pdf/1710.04055.pdf
        """

        # predicting state
        self.Xx = self.__priori_estimate(self.Xx)
        self.Xy = self.__priori_estimate(self.Xy)
        self.Px = self.__estimate_covariance(self.Px)
        self.Py = self.__estimate_covariance(self.Py)
        
        # calculating kalman gain w.r.t both Xx, Xy
        Kx = self.__calculate_kalman_gain(
            self.Px, np.eye(2)*error)
        Ky = self.__calculate_kalman_gain(
            self.Py, np.eye(2)*error)
        
        # No any other relation between velocity and position so filtered out.
        Kx = self.__diagonal_filter(Kx)
        Ky = self.__diagonal_filter(Ky)

        # correcting state estimation using Kalman gain
        self.Xx = self.__postirior_estimate(
            np.array([x1, (x1 - self.old_x)]), Kx, self.Xx)
        self.old_x = x1
        self.Xy = self.__postirior_estimate(
            np.array([y1, (y1 - self.old_y)]), Ky, self.Xy)
        self.old_y = y1

        # correcting covariance matrix using Kalman gain
        self.Px = self.__corrected_covariance(self.Px, Kx)
        self.Py = self.__corrected_covariance(self.Py, Ky)


class kalmanFilterTracker(Kalaman_Filter):
    """ 
    # Object Tracker Class based on kalman filter

    This class keeps not of time_since_update the kalman filter received.
    If kalman filter didnt recieve any update for a long amount of time the 
    object gets destroyed.
    """
    minimum_updates = 7

    def __init__(self, id, dt=1, initial_error=1000, old_x=0, old_y=0, minimum_updates = 7) -> None:
        super().__init__(dt, initial_error, old_x, old_y)
        self.time_since_update = 0
        self.id: int = id
        self.real_updates = 0
        self.minimum_updates = minimum_updates
        self.speed = 0

    def update(self, x1: int, y1: int, error: float):
        super().update(x1, y1, error)
        self.time_since_update = 0
    
    def future_predictions(self, len = 40) -> np.ndarray:
        res = np.zeros(shape=(1, 2))
        if self.real_updates >= self.minimum_updates:
            res = np.zeros(shape=(len, 2))
            i = 1
            X = self.priori_estimate(self.Xx)
            Y = self.priori_estimate(self.Xy)
            res[0][0] = int(X[0])
            res[0][1] = int(Y[0])
            while(i < len):
                X = self.priori_estimate(X)
                Y = self.priori_estimate(Y)
                res[i][0] = int(X[0])
                res[i][1] = int(Y[0])
                i += 1
        return res


class SORT:
    """
    # Maintains the objects in the frame

    Tlost_max = Maximum frame it can fford to loose.

    iou_min = minimum iou between predicted and acctual bounding box
    """
    INITIAL_ID: int = 0
    Tlost_max: int = 3
    iou_min: float = 0.3

    def __init__(self, Tlost_max=3, iou_min=0.3) -> None:
        self.Tlost_max = Tlost_max
        self.iou_min = iou_min
        self.tracker: dict[int, kalmanFilterTracker] = dict()

    def __generate_tracker_predictions(self) -> np.ndarray:
        """ Returns (X_prediction, Y_Prediction, id) """
        k = 0
        predictions = np.zeros((len(self.tracker), 3))
        for i in self.tracker:
            x, y = self.tracker[i].predict_position()
            predictions[k][0] = x
            predictions[k][1] = y
            predictions[k][2] = self.tracker[i].id
            k += 1
        return predictions

    def __generate_id(self) -> int:
        """ Returns new id for the object """
        self.INITIAL_ID += 1
        return self.INITIAL_ID

    def __intersection_over_union(self, boxA, boxB):
        """ Calculated IOU for each model_predicted and tracker_predicted """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def __iou_bulk(self, model_predictions: np.ndarray, tracker_predictions: np.ndarray):
        """ Apply Bulk IOU for every combination of model_predicted and tracker_predicted """
        if len(tracker_predictions) == 0:
            return np.zeros((model_predictions.shape[0], tracker_predictions.shape[0]))

        ret = np.zeros(
            (model_predictions.shape[0], tracker_predictions.shape[0]))
        for i in range(model_predictions.shape[0]):
            for j in range(tracker_predictions.shape[0]):
                x1 = model_predictions[i][0]
                y1 = model_predictions[i][1]
                w = model_predictions[i][2]
                h = model_predictions[i][3]
                x2 = tracker_predictions[j][0]
                y2 = tracker_predictions[j][1]
                box1 = [x1, y1, x1+w, y1+h]
                box2 = [x2, y2, x2+w, y2+h]

                ret[i][j] = self.__intersection_over_union(box1, box2)
        return ret

    def update(self, model_predictions: np.ndarray) -> np.ndarray:
        """
        Input: model_predictions = [[x_center, y_center, mid_width, mid_height, confidence], ...]
        Ouput: [[x_center, y_center, mid_width, mid_height, error_percent, id_assigned, predic_x_centre, predic_y_centre, iou_percet, speed], ...]
        """
        result = np.concatenate((model_predictions.copy(), np.full(
            (model_predictions.shape[0], 5), -1)), axis=1)
        tracker_predictions = self.__generate_tracker_predictions()
        iou_bulk = self.__iou_bulk(model_predictions, tracker_predictions)
        tracked_ids = []

        # IOU Matching
        if iou_bulk != []:
            for i in range(iou_bulk.shape[0]):
                max_iou = iou_bulk[i].argmax()
                if iou_bulk[i][max_iou] >= self.iou_min and result[i][5] == -1:
                    result[i][5] = tracker_predictions[max_iou][2]
                    result[i][6] = tracker_predictions[max_iou][0]
                    result[i][7] = tracker_predictions[max_iou][1]
                    result[i][8] = iou_bulk[i][max_iou]
                    result[i][9] = self.tracker[result[i][5]].speed
                    tracked_ids.append(result[i][5])
                    self.tracker[result[i][5]].real_updates += 1
                    self.tracker[result[i][5]].update(
                                model_predictions[i][0], model_predictions[i][1], (1 - model_predictions[i][4]))

        # creating ID's for not matching kalman filter
        for i in range(len(result)):
            if result[i][5] == -1:
                id = self.__generate_id()
                kf = kalmanFilterTracker(
                    id, old_x=result[i][0], old_y=result[i][1])
                result[i][5] = id
                result[i][6] = result[i][0]
                result[i][7] = result[i][1]
                result[i][8] = 0
                result[i][9] = 0
                self.tracker[result[i][5]] = kf
                tracked_ids.append(result[i][5])

        # Increase the strike and predict kalman filter, Handling excess tracker_predictions
        to_be_deleted = []
        for i in self.tracker:
            if i not in tracked_ids:
                self.tracker[i].update(*self.tracker[i].predict_position(), 0.1)
                if self.tracker[i].time_since_update > self.Tlost_max:
                    to_be_deleted.append(i)

        for i in to_be_deleted:
            del self.tracker[i]

        return result
