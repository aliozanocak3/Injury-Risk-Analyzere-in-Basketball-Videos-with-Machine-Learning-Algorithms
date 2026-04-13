

import pandas as pd
import numpy as np
from src.utils import GeometryUtils, TimeSeriesUtils


class FeatureExtractor:
    
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    
    def __init__(self):
        self.geometry_utils = GeometryUtils()
        self.time_series_utils = TimeSeriesUtils()
        

        self.JUMP_HEIGHT_THRESHOLD = 0.01
        self.LANDING_DECELERATION_THRESHOLD = 0.005
        self.CUTTING_LATERAL_THRESHOLD = 0.01
    
    def extract(self, landmark_csv, output_csv):
        
        df = pd.read_csv(landmark_csv)
        features = []
        
        frames = sorted(df["frame"].unique())
        
        for frame in frames:
            frame_data = df[df["frame"] == frame]
            
            def get_landmark_point(idx):
                landmark_data = frame_data[frame_data["landmark"] == idx]
                if landmark_data.empty:
                    return None
                return (
                    landmark_data["x"].values[0],
                    landmark_data["y"].values[0],
                    landmark_data["z"].values[0] if "z" in landmark_data.columns else 0.0
                )
            
            left_hip = get_landmark_point(self.LEFT_HIP)
            left_knee = get_landmark_point(self.LEFT_KNEE)
            left_ankle = get_landmark_point(self.LEFT_ANKLE)
            
            right_hip = get_landmark_point(self.RIGHT_HIP)
            right_knee = get_landmark_point(self.RIGHT_KNEE)
            right_ankle = get_landmark_point(self.RIGHT_ANKLE)
            
            feature_dict = {"frame": frame}
            
            if left_hip and left_knee and left_ankle:
                feature_dict["left_knee_angle"] = self.geometry_utils.calculate_angle(
                    left_hip[:2], left_knee[:2], left_ankle[:2]
                )
                if right_knee:
                    feature_dict["left_hip_angle"] = self.geometry_utils.calculate_angle(
                        left_knee[:2], left_hip[:2], right_knee[:2]
                    )
                else:
                    feature_dict["left_hip_angle"] = np.nan
                feature_dict["left_ankle_angle"] = self.geometry_utils.calculate_angle(
                    left_knee[:2], left_ankle[:2], (left_ankle[0], left_ankle[1] + 0.1)
                )
            else:
                feature_dict["left_knee_angle"] = np.nan
                feature_dict["left_hip_angle"] = np.nan
                feature_dict["left_ankle_angle"] = np.nan
            
            
            if right_hip and right_knee and right_ankle:
                feature_dict["right_knee_angle"] = self.geometry_utils.calculate_angle(
                    right_hip[:2], right_knee[:2], right_ankle[:2]
                )
                if left_knee:
                    feature_dict["right_hip_angle"] = self.geometry_utils.calculate_angle(
                        right_knee[:2], right_hip[:2], left_knee[:2]
                    )
                else:
                    feature_dict["right_hip_angle"] = np.nan
                feature_dict["right_ankle_angle"] = self.geometry_utils.calculate_angle(
                    right_knee[:2], right_ankle[:2], (right_ankle[0], right_ankle[1] + 0.1)
                )
            else:
                feature_dict["right_knee_angle"] = np.nan
                feature_dict["right_hip_angle"] = np.nan
                feature_dict["right_ankle_angle"] = np.nan
            
            if not np.isnan(feature_dict.get("left_knee_angle", np.nan)) and \
               not np.isnan(feature_dict.get("right_knee_angle", np.nan)):
                feature_dict["knee_angle"] = (
                    feature_dict["left_knee_angle"] + feature_dict["right_knee_angle"]
                ) / 2.0
            elif not np.isnan(feature_dict.get("left_knee_angle", np.nan)):
                feature_dict["knee_angle"] = feature_dict["left_knee_angle"]
            elif not np.isnan(feature_dict.get("right_knee_angle", np.nan)):
                feature_dict["knee_angle"] = feature_dict["right_knee_angle"]
            else:
                feature_dict["knee_angle"] = np.nan

            if not np.isnan(feature_dict.get("left_hip_angle", np.nan)) and \
               not np.isnan(feature_dict.get("right_hip_angle", np.nan)):
                feature_dict["hip_angle"] = (
                    feature_dict["left_hip_angle"] + feature_dict["right_hip_angle"]
                ) / 2.0
            elif not np.isnan(feature_dict.get("left_hip_angle", np.nan)):
                feature_dict["hip_angle"] = feature_dict["left_hip_angle"]
            elif not np.isnan(feature_dict.get("right_hip_angle", np.nan)):
                feature_dict["hip_angle"] = feature_dict["right_hip_angle"]
            else:
                feature_dict["hip_angle"] = np.nan
            
            if not np.isnan(feature_dict.get("left_ankle_angle", np.nan)) and \
               not np.isnan(feature_dict.get("right_ankle_angle", np.nan)):
                feature_dict["ankle_angle"] = (
                    feature_dict["left_ankle_angle"] + feature_dict["right_ankle_angle"]
                ) / 2.0
            elif not np.isnan(feature_dict.get("left_ankle_angle", np.nan)):
                feature_dict["ankle_angle"] = feature_dict["left_ankle_angle"]
            elif not np.isnan(feature_dict.get("right_ankle_angle", np.nan)):
                feature_dict["ankle_angle"] = feature_dict["right_ankle_angle"]
            else:
                feature_dict["ankle_angle"] = np.nan
            

            if left_hip and left_ankle:
                feature_dict["hip_y"] = left_hip[1]
                feature_dict["ankle_y"] = left_ankle[1]
                feature_dict["hip_x"] = left_hip[0]
                feature_dict["ankle_x"] = left_ankle[0]
            elif right_hip and right_ankle:
                feature_dict["hip_y"] = right_hip[1]
                feature_dict["ankle_y"] = right_ankle[1]
                feature_dict["hip_x"] = right_hip[0]
                feature_dict["ankle_x"] = right_ankle[0]
            else:
                feature_dict["hip_y"] = np.nan
                feature_dict["ankle_y"] = np.nan
                feature_dict["hip_x"] = np.nan
                feature_dict["ankle_x"] = np.nan
            
            features.append(feature_dict)
        
        out = pd.DataFrame(features)
        
        if len(out) > 1:
            out["knee_angular_velocity"] = self.time_series_utils.angular_velocity(
                out["knee_angle"].ffill().bfill()
            )
            out["hip_angular_velocity"] = self.time_series_utils.angular_velocity(
                out["hip_angle"].ffill().bfill()
            )
            out["ankle_angular_velocity"] = self.time_series_utils.angular_velocity(
                out["ankle_angle"].ffill().bfill()
            )
        else:
            out["knee_angular_velocity"] = 0.0
            out["hip_angular_velocity"] = 0.0
            out["ankle_angular_velocity"] = 0.0
        

        if len(out) > 1:
            hip_y_filled = out["hip_y"].ffill().bfill()
            ankle_y_filled = out["ankle_y"].ffill().bfill()
            hip_x_filled = out["hip_x"].ffill().bfill()
            ankle_x_filled = out["ankle_x"].ffill().bfill()
            out["hip_velocity_y"] = -hip_y_filled.diff()
            out["ankle_velocity_y"] = -ankle_y_filled.diff()

            out["hip_velocity_x"] = hip_x_filled.diff()
            out["ankle_velocity_x"] = ankle_x_filled.diff()

            out["hip_acceleration_y"] = out["hip_velocity_y"].diff()
            out["ankle_acceleration_y"] = out["ankle_velocity_y"].diff()

            out["hip_velocity_y"] = out["hip_velocity_y"].fillna(0.0)
            out["ankle_velocity_y"] = out["ankle_velocity_y"].fillna(0.0)
            out["hip_velocity_x"] = out["hip_velocity_x"].fillna(0.0)
            out["ankle_velocity_x"] = out["ankle_velocity_x"].fillna(0.0)
            out["hip_acceleration_y"] = out["hip_acceleration_y"].fillna(0.0)
            out["ankle_acceleration_y"] = out["ankle_acceleration_y"].fillna(0.0)
        else:
            out["hip_velocity_y"] = 0.0
            out["ankle_velocity_y"] = 0.0
            out["hip_velocity_x"] = 0.0
            out["ankle_velocity_x"] = 0.0
            out["hip_acceleration_y"] = 0.0
            out["ankle_acceleration_y"] = 0.0
        out = self._detect_movement_patterns(out)
        
        out.to_csv(output_csv, index=False)
        return out
    
    def _detect_movement_patterns(self, df):
        df["is_jump"] = 0
        df["is_landing"] = 0
        df["is_cutting"] = 0
        df["jump_height"] = 0.0
        df["landing_force_estimate"] = 0.0
        df["cutting_angle"] = 0.0
        
        if len(df) < 3:
            return df

        hip_y_filled = df["hip_y"].ffill().bfill()
        if not hip_y_filled.isna().all():
            baseline_hip_y = hip_y_filled.quantile(0.1)
            peak_hip_y = hip_y_filled.min()
            df["jump_height"] = baseline_hip_y - peak_hip_y
        
        for i in range(1, len(df)):
            if not pd.isna(df.iloc[i]["hip_y"]) and not pd.isna(df.iloc[i-1]["hip_y"]):
                hip_y_diff = df.iloc[i-1]["hip_y"] - df.iloc[i]["hip_y"]
                ankle_y_diff = 0.0
                
                if not pd.isna(df.iloc[i]["ankle_y"]) and not pd.isna(df.iloc[i-1]["ankle_y"]):
                    ankle_y_diff = df.iloc[i-1]["ankle_y"] - df.iloc[i]["ankle_y"]

                if hip_y_diff > self.JUMP_HEIGHT_THRESHOLD or ankle_y_diff > self.JUMP_HEIGHT_THRESHOLD:
                    df.iloc[i, df.columns.get_loc("is_jump")] = 1
        
        for i in range(1, len(df)):
            if not pd.isna(df.iloc[i]["ankle_y"]) and not pd.isna(df.iloc[i-1]["ankle_y"]):
                ankle_y_diff = df.iloc[i]["ankle_y"] - df.iloc[i-1]["ankle_y"]
                
                if ankle_y_diff > self.LANDING_DECELERATION_THRESHOLD:
                    knee_angle_ok = True
                    if not pd.isna(df.iloc[i]["knee_angle"]):
                        knee_angle_ok = df.iloc[i]["knee_angle"] < 150
                    
                    if knee_angle_ok:
                        df.iloc[i, df.columns.get_loc("is_landing")] = 1

                        if not pd.isna(df.iloc[i]["ankle_acceleration_y"]):
                            df.iloc[i, df.columns.get_loc("landing_force_estimate")] = abs(
                                df.iloc[i]["ankle_acceleration_y"]
                            )
        
        for i in range(2, len(df)):
            lateral_velocity = abs(df.iloc[i]["hip_velocity_x"])
            if lateral_velocity > self.CUTTING_LATERAL_THRESHOLD:
                if i > 1:
                    prev_velocity_x = df.iloc[i-1]["hip_velocity_x"]
                    curr_velocity_x = df.iloc[i]["hip_velocity_x"]

                    if (prev_velocity_x * curr_velocity_x < 0 or
                        abs(curr_velocity_x - prev_velocity_x) > self.CUTTING_LATERAL_THRESHOLD):
                        df.iloc[i, df.columns.get_loc("is_cutting")] = 1

                        if not pd.isna(df.iloc[i]["hip_velocity_x"]) and not pd.isna(df.iloc[i]["hip_velocity_y"]):
                            cutting_angle = np.degrees(np.arctan2(
                                abs(df.iloc[i]["hip_velocity_x"]),
                                abs(df.iloc[i]["hip_velocity_y"])
                            ))
                            df.iloc[i, df.columns.get_loc("cutting_angle")] = cutting_angle
        
        df["knee_angle_during_landing"] = np.where(
            df["is_landing"] == 1,
            df["knee_angle"],
            np.nan
        )
        
        if "left_knee_angle" in df.columns and "right_knee_angle" in df.columns:
            df["knee_angle_asymmetry"] = abs(
                df["left_knee_angle"].fillna(0) - df["right_knee_angle"].fillna(0)
            )
            df["asymmetry_during_cutting"] = np.where(
                df["is_cutting"] == 1,
                df["knee_angle_asymmetry"],
                np.nan
            )
        
        return df
