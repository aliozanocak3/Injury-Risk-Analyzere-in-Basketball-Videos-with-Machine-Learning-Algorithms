

import numpy as np


class GeometryUtils:
    
    @staticmethod
    def calculate_angle(point_a, point_b, point_c):
        a = np.array(point_a)
        b = np.array(point_b)
        c = np.array(point_c)
        
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return np.nan
        
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        
        cosine = np.clip(cosine, -1.0, 1.0)
        
        return np.degrees(np.arccos(cosine))


class TimeSeriesUtils:
    
    @staticmethod
    def angular_velocity(angle_values, dt=1.0):
        values = np.array(angle_values)
        
        return np.gradient(values, dt)
    
    @staticmethod
    def smooth_signal(values, window_size=5):
        values = np.array(values)
        
        if len(values) < window_size:
            return values
        
        if window_size % 2 == 0:
            window_size += 1
        
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(values, kernel, mode='same')
        
        return smoothed
