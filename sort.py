# Dependancies
import numpy as np
from typing import Tuple
from math import sqrt, pow


class KalmanFilter:
    """Predict Future (X_Cord, Y_Cord) based on input."""

    def __init__(self, dt=1, initial_error=1000, old_input_x=0, old_input_y=0):
        """
        dt: is delta time between the periodic readings.
        initial_error: Inital error 
        old_input_x, old_input_y: inital input for x, y

        # Understanding Initial Mat Matrix
        ------------------------------------------------------------------------
        [0, 0],					=> X state matrix 				=> Mat[0, :]
        [1, dt],			a	=> A state transition matrix	=> Mat[1:3, :]
        [0, 1],
        [initial_error, 0],		=> covariance matrix for X		=> Mat[3:5, :]
        [0, initial_error],
        [0, 0],					=> Y state matrix				=> Mat[5, :]
        [initial_error, 0],		=> covariance matrix for Y		=> Mat[6:, :]
        [0, initial_error]
        ------------------------------------------------------------------------
        """
        self.dt = dt
        self.old_input_x = old_input_x
        self.old_input_y = old_input_y
        self.Mat = np.array([
            [0, 0],
            [1, dt],
            [0, 1],
            [initial_error, 0],
            [0, initial_error],
            [0, 0],
            [initial_error, 0],
            [0, initial_error]
        ], dtype=np.float16)

    def predict_distance(self) -> Tuple[int, int]:
        """Predicts Future State With Recorded Readings But it doesnt update itself.
        Returns: [X_Distance, Y_Distance]
        """
        X = int(self.Mat[1:3, :].dot(self.Mat[0, :])[0])
        Y = int(self.Mat[1:3, :].dot(self.Mat[5, :])[0])
        return (X, Y)

    def predict_velocity(self) -> np.float16:
        """Merge's Both X_Velocity and Y_Velocity together.

        Returns: [Velocity]
        """
        vx, vy = self.predict_2d_velocity()
        return sqrt(pow(vx, 2) + pow(vy, 2))

    def predict_2d_velocity(self) -> Tuple[np.float16, np.float16]:
        """Predicts Future State With Recorded Readings But it doesnt update itself.

        Returns: [X_Velocity, Y_Velocity]
        """
        X = self.Mat[1:3, :].dot(self.Mat[0, :])[1]
        Y = self.Mat[1:3, :].dot(self.Mat[5, :])[1]
        return (X, Y)

    def update(self, input_x: int, input_y: int, error: float) -> None:
        """Updates kalman filter with new data and it doesn't predicts."""

        self.Mat[0, :] = self.Mat[1:3, :].dot(self.Mat[0, :])
        self.Mat[5, :] = self.Mat[1:3, :].dot(self.Mat[5, :])
        self.Mat[3:5, :] = self.Mat[1:3, :].dot(
            self.Mat[3:5, :].dot(self.Mat[1:3, :]))
        self.Mat[6:, :] = self.Mat[1:3, :].dot(
            self.Mat[6:, :].dot(self.Mat[1:3, :]))
        R = np.eye(2)*error
        Kx = np.divide(self.Mat[3:5, :], (self.Mat[3:5, :]+R),
                       where=(self.Mat[3:5, :]+R) != 0)
        Ky = np.divide(self.Mat[6:, :], (self.Mat[6:, :]+R),
                       where=(self.Mat[6:, :]+R) != 0)
        Yx = np.array([[input_x], [(input_x - self.old_input_x)/self.dt]])
        Yy = np.array([[input_y], [(input_y - self.old_input_y)/self.dt]])
        self.old_input_x = input_x
        self.old_input_y = input_y
        self.Mat[0, :] = self.Mat[0, :] + \
            Kx.dot(Yx - self.Mat[0, :].reshape(2, 1)).reshape(1, 2)
        self.Mat[5, :] = self.Mat[5, :] + \
            Ky.dot(Yy - self.Mat[5, :].reshape(2, 1)).reshape(1, 2)
        self.Mat[3:5, :] = (np.eye(2) - Kx).dot(self.Mat[3:5, :])
        self.Mat[6:, :] = (np.eye(2) - Ky).dot(self.Mat[6:, :])

    def forward_position(self, level: int) -> np.ndarray:
        ret = np.zeros(shape=(level, 2))
        sx = self.Mat[0, :].copy()
        sy = self.Mat[5, :].copy()
        for i in ret:
            sx = self.Mat[1:3, :].dot(sx)
            sy = self.Mat[1:3, :].dot(sy)
            i[0] = sx[0]
            i[1] = sy[0]
        return ret


class OBJ(KalmanFilter):
    INITIAL_OBJECT_ID = 1

    def __init__(self, dt=1, initial_error=1000, old_input_x=0, old_input_y=0):
        super().__init__(dt, initial_error, old_input_x, old_input_y)
        self.id = OBJ.INITIAL_OBJECT_ID
        self.time_passed_after = 0
        self.assigned = False
        OBJ.INITIAL_OBJECT_ID += 1
        self.speed = 0


class SORT():
    pass


if __name__ == "__main__":
    kf = KalmanFilter()
    kf.update(1, 1, 0.6)
    kf.update(2, 2, 0.6)
    kf.update(10, 10, 10000)
    kf.update(4, 4, 0.6)
    kf.update(5, 5, 0.6)
    print(kf.predict_distance())
    pass