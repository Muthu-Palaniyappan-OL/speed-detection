"""
Implementation Of kalman Filter

Based on constant velocity model.

Using:
    * numpy

Usage: 
    =========================== 1D Kalman Filter =============================
        Kf = KalmanFilter_1D()
        Kf.update(10, 0.1) # 0.1 percentage error and measurment value 10  
        Kf.update(15, 0.1)
        Kf.update(20, 0.1)
        print(Kf.predict())                                                
    ==========================================================================

"""
import numpy as np
from typing import Tuple

#####################
MIN_UPDATE_TIMES = 5
#####################

class KalmanFilter:
    def __init__(self, dt = 1, initial_error = 1000, old_input = 0):
        """
        dt is delta time between the readings (supposed to be at regular intervals).
        initial_error is within what range measurement value (if your expecting measurment value within 1000 put 1000, anything more than that also no problem).
        old_input if measurement doesnt start in 0 position change old_input value. even if you don't change it doesn't matter much.
        """
        self.update_count = 0
        self.kf_x = KalmanFilter_1D(dt, initial_error, old_input)
        self.kf_y = KalmanFilter_1D(dt, initial_error, old_input)
    
    def predict(self) -> Tuple[int, float, int, float]:
        """
        Predicts Future State With Recorded Readings But it doesnt update itself
        """
        return self.kf_x.predict(), self.kf_y.predict()
        
    def update(self, input_x :int, input_y :int, error :float):
        """
        Updates kalman filter with new data and it doesn't predicts
        """
        self.kf_x.update(input_x, error)
        self.kf_y.update(input_y, error)
        self.update_count += 1
    
    def future(self, times :int) -> Tuple[int, float, int, float]:
        """
        Predicts data in future after some "times".
        KalmanFilter.predict() is equivalent to KalmanFilter.future(1)
        """
        return self.kf_x.update(times), self.kf_y.update(times)
    
    def worthy_enough(self) -> bool:
        return self.update_count >= MIN_UPDATE_TIMES
    
    def forward(self):
        if self.worthy_enough() is True:
            pre = self.kf_x.predict()
            self.kf_x.update(pre[0], pre[2])

class KalmanFilter_1D:
    def __init__(self, dt = 1, initial_error = 1000, old_input = 0):
        """
        dt is delta time between the readings (supposed to be at regular intervals).
        initial_error is within what range measurement value (if your expecting measurment value within 1000 put 1000, anything more than that also no problem).
        old_input if measurement doesnt start in 0 position change old_input value. even if you don't change it doesn't matter much.
        """
        self.dt = dt
        self.old_input = old_input
        self.update_count = 0
        self.X = np.array([[0],[0]])
        self.A = np.array([
            [1, self.dt],
            [0, 1]
        ])
        self.P = np.eye(2)*initial_error
    
    def predict(self) -> Tuple[int, float]:
        """
        Predicts Future State With Recorded Readings But it doesnt update itself
        """
        return self.A.dot(self.X)
    
    def worthy_enough(self) -> bool:
        return self.update_count >= MIN_UPDATE_TIMES
        
    def update(self, input :int, error :float):
        """
        Updates kalman filter with new data and it doesn't predicts
        """
        self.X = self.A.dot(self.X)
        self.P = self.A.dot(self.P.dot(self.A))
        R = np.eye(2)*error
        K = np.divide(self.P, (self.P+R), where=(self.P+R)!=0)
        Y = np.array([[input], [(input - self.old_input)/self.dt]])
        self.old_input = input
        X = self.X + K.dot(Y - self.X)
        self.X = X
        self.P = (np.eye(2) - K).dot(self.P)
        self.update_count += 1
    
    def future(self, times :int) -> Tuple[int, float]:
        """
        Predicts data in future after some "times".
        KalmanFilter_1D.predict() is equivalent to KalmanFilter_1D.future(1)
        """
        
        s = self.X
        while(times):
            s = self.A.dot(s)
            times -= 1
        
        return s

if __name__ == "__main__":
    KF = KalmanFilter()
    print(KF.predict())
    KF.update(10, 20, 0.1)
    print(KF.predict())
    KF.update(15, 25, 0.1)
    print(KF.predict())
    KF.update(20, 30, 0.1)
    print(KF.predict())
    KF.update(25, 35, 0.1)
    print(KF.predict())
    KF.update(30, 40, 0.1)
    print(KF.predict())