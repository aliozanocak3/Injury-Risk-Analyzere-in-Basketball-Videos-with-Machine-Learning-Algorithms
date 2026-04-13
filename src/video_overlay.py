

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from src.utils import GeometryUtils, TimeSeriesUtils
from src.calculate_features import FeatureExtractor
from src.ml_models.base_model import BaseModel
import joblib


class VideoAnalyzer:
   
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    
    RISK_COLORS = {
        "Low": (0, 255, 0),      
        "Medium": (0, 165, 255),  
        "High": (0, 0, 255)       
    }
    
   
    LOWER_EXTREMITY_LANDMARKS = [
        23, 24,  # Left hip, Right hip
        25, 26,  # Left knee, Right knee
        27, 28,  # Left ankle, Right ankle
    ]
    
   
    LOWER_EXTREMITY_CONNECTIONS = [
        (23, 25),  # Left hip to left knee
        (25, 27),  # Left knee to left ankle
        (24, 26),  # Right hip to right knee
        (26, 28),  # Right knee to right ankle
    ]
    
    def __init__(self, model_path, model_type="random_forest"):
       
        self.model = self._load_model(model_path, model_type)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.geometry_utils = GeometryUtils()
        self.feature_extractor = FeatureExtractor()
        
        self.frame_data = []
    
    def _load_model(self, model_path, model_type):
        """Load trained model from disk."""
       
        if model_type == "random_forest":
            from src.ml_models.random_forest_model import RandomForestModel
            model = RandomForestModel()
        elif model_type == "svm":
            from src.ml_models.svm_model import SVMModel
            model = SVMModel()
        elif model_type == "xgboost":
            from src.ml_models.xgboost_model import XGBoostModel
            model = XGBoostModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load(model_path)
        return model
    
    def analyze(self, video_path, output_path):
       
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            
            features = None
            risk_label = "Unknown"
            risk_color = (128, 128, 128)  
            
            if results.pose_landmarks:
                
                features = self._extract_frame_features(
                    results.pose_landmarks.landmark, w, h
                )
                
                
                current_risk_label = "Unknown"
                current_risk_color = (128, 128, 128)  
                
                if features is not None:
                    
                    feature_list = [
                        features["knee_angle"],
                        features.get("knee_angular_velocity", 0.0),
                        features.get("hip_angle", 0.0),
                        features.get("ankle_angle", 0.0)
                    ]
                    
                   
                    if "is_jump" in features:
                        feature_list.append(float(features["is_jump"]))
                    else:
                        feature_list.append(0.0)
                    
                    if "is_landing" in features:
                        feature_list.append(float(features["is_landing"]))
                    else:
                        feature_list.append(0.0)
                    
                    if "is_cutting" in features:
                        feature_list.append(float(features["is_cutting"]))
                    else:
                        feature_list.append(0.0)
                    
                   
                    if "is_landing" in features and features["is_landing"] == 1:
                        feature_list.append(features.get("knee_angle", 0.0))  
                    else:
                        feature_list.append(0.0)
                    
                    
                    if "is_cutting" in features and features["is_cutting"] == 1:
                        left_angle = features.get("left_knee_angle", features.get("knee_angle", 0))
                        right_angle = features.get("right_knee_angle", features.get("knee_angle", 0))
                        asymmetry = abs(left_angle - right_angle)
                        feature_list.append(asymmetry)
                    else:
                        feature_list.append(0.0)
                    
                    
                    feature_list.append(features.get("landing_force_estimate", 0.0))
                    feature_list.append(features.get("jump_height", 0.0))
                    feature_list.append(features.get("cutting_angle", 0.0))
                    
                   
                    feature_list.append(features.get("hip_velocity_y", 0.0))
                    feature_list.append(features.get("ankle_velocity_y", 0.0))
                    feature_list.append(features.get("ankle_acceleration_y", 0.0))
                    
                    feature_vector = np.array([feature_list])
                    
                    
                    prediction = self.model.predict(feature_vector)[0]
                    
                    
                    prediction_proba = None
                    if hasattr(self.model.model, 'predict_proba'):
                        try:
                            proba = self.model.model.predict_proba(feature_vector)[0]
                            prediction_proba = proba
                        except:
                            pass
                    
                   
                    if hasattr(self.model.model, 'classes_'):
                        classes = self.model.model.classes_
                       
                        if len(classes) > 0 and isinstance(classes[0], (np.integer, int)):
                           
                            if len(classes) == 2:
                               
                                label_map = {0: "High Risk", 1: "Non-High Risk"}
                            else:
                                
                                label_map = {0: "Low", 1: "Medium", 2: "High"}
                        else:
                            
                            label_map = {i: label for i, label in enumerate(classes)}
                        risk_label = label_map.get(prediction, "Unknown")
                    else:
                       
                        if hasattr(self.model.model, 'n_classes_'):
                            if self.model.model.n_classes_ == 2:
                                
                                label_map = {0: "High Risk", 1: "Non-High Risk"}
                            else:
                                
                                label_map = {0: "Low", 1: "Medium", 2: "High"}
                        else:
                            label_map = {0: "Low", 1: "Medium", 2: "High"}
                        risk_label = label_map.get(prediction, "Unknown")
                    
                    
                    risk_label_str = str(risk_label)
                    
                   
                    if risk_label_str == "High Risk" or risk_label == 0:
                        display_label = "High"
                    elif risk_label_str == "Non-High Risk" or risk_label == 1:
                        if prediction_proba is not None and len(prediction_proba) == 2:
                            
                            non_high_risk_prob = prediction_proba[1]  
                            
                            if non_high_risk_prob > 0.85:
                                display_label = "Low"
                            elif non_high_risk_prob > 0.6:
                                display_label = "Medium"
                            else:
                                
                                if features.get("knee_angle", 180) > 90 and \
                                   features.get("is_landing", 0) == 0 and \
                                   features.get("is_cutting", 0) == 0:
                                    display_label = "Low"
                                else:
                                    display_label = "Medium"
                        else:
                            
                            knee_angle = features.get("knee_angle", 180)
                            is_landing = features.get("is_landing", 0) == 1
                            is_cutting = features.get("is_cutting", 0) == 1
                            
                            
                            if knee_angle > 90 and not is_landing and not is_cutting:
                                display_label = "Low"
                            
                            elif knee_angle > 60 or is_landing or is_cutting:
                                display_label = "Medium"
                            else:
                                
                                display_label = "Medium"
                    elif risk_label_str in ["High", "Medium", "Low"]:
                        display_label = risk_label_str
                    else:
                        
                        if isinstance(prediction, (int, np.integer)):
                            if prediction == 0:
                                display_label = "High"
                            else:
                                
                                knee_angle = features.get("knee_angle", 180)
                                is_landing = features.get("is_landing", 0) == 1
                                is_cutting = features.get("is_cutting", 0) == 1
                                
                                if knee_angle > 90 and not is_landing and not is_cutting:
                                    display_label = "Low"
                                else:
                                    display_label = "Medium"
                        else:
                            display_label = "Low"  
                    
                    risk_color = self.RISK_COLORS.get(display_label, (128, 128, 128))
                    risk_label = display_label  
                    current_risk_label = risk_label
                    current_risk_color = risk_color
                    
                    
                    frame_data.append({
                        "frame": frame_idx,
                        "knee_angle": features["knee_angle"],
                        "hip_angle": features.get("hip_angle", np.nan),
                        "ankle_angle": features.get("ankle_angle", np.nan),
                        "risk_label": risk_label
                    })
                
                
                self._draw_pose_with_risk_coloring(
                    frame, 
                    results.pose_landmarks, 
                    w, h,
                    current_risk_label,
                    current_risk_color
                )
            

            self._overlay_risk_annotation(frame, current_risk_label, current_risk_color)
            
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        self.frame_data = pd.DataFrame(frame_data)
        
        return {
            "total_frames": frame_idx,
            "frame_data": self.frame_data,
            "output_path": output_path
        }
    
    def _extract_frame_features(self, landmarks, width, height):
       
        def get_landmark_point(idx):
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            return (lm.x * width, lm.y * height)
        
        left_hip = get_landmark_point(self.LEFT_HIP)
        left_knee = get_landmark_point(self.LEFT_KNEE)
        left_ankle = get_landmark_point(self.LEFT_ANKLE)
        right_hip = get_landmark_point(self.RIGHT_HIP)
        right_knee = get_landmark_point(self.RIGHT_KNEE)
        right_ankle = get_landmark_point(self.RIGHT_ANKLE)
        
        features = {}
        
        left_knee_angle = None
        if left_hip and left_knee and left_ankle:
            left_knee_angle = self.geometry_utils.calculate_angle(
                left_hip, left_knee, left_ankle
            )
        
        right_knee_angle = None
        if right_hip and right_knee and right_ankle:
            right_knee_angle = self.geometry_utils.calculate_angle(
                right_hip, right_knee, right_ankle
            )
        
        if left_knee_angle is not None and right_knee_angle is not None:
            features["knee_angle"] = (left_knee_angle + right_knee_angle) / 2.0
            features["left_knee_angle"] = left_knee_angle
            features["right_knee_angle"] = right_knee_angle
        elif left_knee_angle is not None:
            features["knee_angle"] = left_knee_angle
            features["left_knee_angle"] = left_knee_angle
            features["right_knee_angle"] = left_knee_angle  
        elif right_knee_angle is not None:
            features["knee_angle"] = right_knee_angle
            features["left_knee_angle"] = right_knee_angle  
            features["right_knee_angle"] = right_knee_angle
        else:
            return None  
        
        if left_hip and left_knee and right_knee:
            features["hip_angle"] = self.geometry_utils.calculate_angle(
                left_knee, left_hip, right_knee
            )
        elif right_hip and right_knee and left_knee:
            features["hip_angle"] = self.geometry_utils.calculate_angle(
                right_knee, right_hip, left_knee
            )
        
        if left_knee and left_ankle:
            vertical_ref = (left_ankle[0], left_ankle[1] + 0.1 * height)
            features["ankle_angle"] = self.geometry_utils.calculate_angle(
                left_knee, left_ankle, vertical_ref
            )
        elif right_knee and right_ankle:
            vertical_ref = (right_ankle[0], right_ankle[1] + 0.1 * height)
            features["ankle_angle"] = self.geometry_utils.calculate_angle(
                right_knee, right_ankle, vertical_ref
            )
        
        if hasattr(self, '_prev_knee_angle'):
            features["knee_angular_velocity"] = features["knee_angle"] - self._prev_knee_angle
        else:
            features["knee_angular_velocity"] = 0.0
        
        if not hasattr(self, '_prev_frame_data'):
            self._prev_frame_data = {}
        
        if left_hip and left_ankle:
            features["hip_y"] = left_hip[1]
            features["ankle_y"] = left_ankle[1]
            features["hip_x"] = left_hip[0]
            features["ankle_x"] = left_ankle[0]
            
            if "prev_hip_y" in self._prev_frame_data:
                features["hip_velocity_y"] = -(features["hip_y"] - self._prev_frame_data["prev_hip_y"])
                features["ankle_velocity_y"] = -(features["ankle_y"] - self._prev_frame_data["prev_ankle_y"])
            else:
                features["hip_velocity_y"] = 0.0
                features["ankle_velocity_y"] = 0.0
            
            features["is_jump"] = 0
            features["is_landing"] = 0
            features["is_cutting"] = 0
            
            if features["hip_velocity_y"] < -0.05 or features["ankle_velocity_y"] < -0.05:
                features["is_jump"] = 1
            
            if (features["ankle_velocity_y"] > 0.02 and 
                features["knee_angle"] < 90):
                features["is_landing"] = 1
            
            if "prev_hip_x" in self._prev_frame_data:
                lateral_velocity = abs(features["hip_x"] - self._prev_frame_data["prev_hip_x"])
                if lateral_velocity > 0.03:
                    features["is_cutting"] = 1
            
            self._prev_frame_data["prev_hip_y"] = features["hip_y"]
            self._prev_frame_data["prev_ankle_y"] = features["ankle_y"]
            self._prev_frame_data["prev_hip_x"] = features["hip_x"]
        else:
            features["hip_y"] = np.nan
            features["ankle_y"] = np.nan
            features["hip_x"] = np.nan
            features["ankle_x"] = np.nan
            features["hip_velocity_y"] = 0.0
            features["ankle_velocity_y"] = 0.0
            features["is_jump"] = 0
            features["is_landing"] = 0
            features["is_cutting"] = 0
        
        self._prev_knee_angle = features["knee_angle"]
        
        return features
    
    def _draw_pose_with_risk_coloring(self, frame, pose_landmarks, width, height, risk_label, risk_color):
        """Draw pose skeleton with risk-based coloring for lower extremity."""
        landmarks = pose_landmarks.landmark
        
        default_left_color = (255, 144, 30)   # Orange (BGR)
        default_right_color = (0, 217, 231)    # Cyan (BGR)
        
        if risk_label == "High":
            lower_extremity_color = (0, 0, 255)  # Red (BGR)
        elif risk_label == "Medium":
            lower_extremity_color = (0, 165, 255)  # Orange (BGR)
        else:  
            lower_extremity_color = (0, 255, 0)  # Green (BGR)
        
        upper_body_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            (11, 23), (12, 24),
        ]
        
        for connection in upper_body_connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                if start_idx in [11, 13, 15, 23] or end_idx in [11, 13, 15, 23]:
                    color = default_left_color
                elif start_idx in [12, 14, 16, 24] or end_idx in [12, 14, 16, 24]:
                    color = default_right_color
                else:
                    color = default_left_color  
                
                start_point = (int(start.x * width), int(start.y * height))
                end_point = (int(end.x * width), int(end.y * height))
                
                cv2.line(frame, start_point, end_point, color, 2)
        
        for connection in self.LOWER_EXTREMITY_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_point = (int(start.x * width), int(start.y * height))
                end_point = (int(end.x * width), int(end.y * height))
                
                cv2.line(frame, start_point, end_point, lower_extremity_color, 3)
        
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            if idx in self.LOWER_EXTREMITY_LANDMARKS:   
                cv2.circle(frame, (x, y), 5, lower_extremity_color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)  
            else:
                if idx in [11, 13, 15, 23, 25, 27]:  
                    cv2.circle(frame, (x, y), 5, default_left_color, -1)
                elif idx in [12, 14, 16, 24, 26, 28]:  
                    cv2.circle(frame, (x, y), 5, default_right_color, -1)
                else:
                    cv2.circle(frame, (x, y), 5, default_left_color, -1)
    
    def _overlay_risk_annotation(self, frame, risk_label, color):
        """Overlay risk annotation with improved visibility."""
        if risk_label == "High":
            text = "⚠️ HIGH RISK"
            bg_color = (0, 0, 255)  
            text_color = (255, 255, 255)  
        elif risk_label == "Medium":
            text = "⚡ MEDIUM RISK"
            bg_color = (0, 165, 255)  
            text_color = (255, 255, 255)  
        elif risk_label == "Low":
            text = "✓ LOW RISK"
            bg_color = (0, 255, 0)  
            text_color = (0, 0, 0)  
        else:
            text = "? UNKNOWN"
            bg_color = (128, 128, 128)  
            text_color = (255, 255, 255)  
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3  
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        padding = 15
        rect_x1 = 20
        rect_y1 = 20
        rect_x2 = rect_x1 + text_width + padding * 2
        rect_y2 = rect_y1 + text_height + padding * 2
        
        cv2.rectangle(
            frame,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            bg_color,
            -1
        )
        
        cv2.rectangle(
            frame,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            (255, 255, 255),  
            2
        )
        
        text_x = rect_x1 + padding
        text_y = rect_y1 + text_height + padding - 5
        
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
