import numpy as np


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def update_error(self, cte):
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte

    def total_error(self):
        return (-self.Kp * self.p_error) - (self.Ki * self.i_error) - (self.Kd * self.d_error)