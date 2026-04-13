

import pandas as pd
import numpy as np


class RiskLabeler:
    
    KNEE_ANGLE_HIGH_RISK_THRESHOLD = 30.0  
    KNEE_ANGLE_MEDIUM_RISK_THRESHOLD = 60.0  
    
    ANGULAR_VELOCITY_HIGH_THRESHOLD = 200.0  
    
    HIP_ANGLE_OPTIMAL_MIN = 120.0  
    HIP_ANGLE_OPTIMAL_MAX = 180.0  
    
    def __init__(self):
        pass
    
    def label(self, feature_csv, output_csv, binary_classification=False):
        df = pd.read_csv(feature_csv)

        risk_scores = []
        
        for idx, row in df.iterrows():
            score = self._calculate_risk_score(row)
            risk_scores.append(score)
        
        df["risk_label"] = [self._score_to_label(score) for score in risk_scores]
        df["risk_score"] = risk_scores  
        
        if binary_classification:
            df["risk_label"] = df["risk_label"].apply(self._to_binary_label)
        
        df.to_csv(output_csv, index=False)
        return df
    
    def _to_binary_label(self, label):
        if label == "High":
            return "High Risk"
        else:  
            return "Non-High Risk"
    
    def _calculate_risk_score(self, row):
        score = 0.0
        factors = 0

        is_jump = row.get("is_jump", 0) == 1
        is_landing = row.get("is_landing", 0) == 1
        is_cutting = row.get("is_cutting", 0) == 1

        knee_angle = row.get("knee_angle")
        left_knee_angle = row.get("left_knee_angle")
        right_knee_angle = row.get("right_knee_angle")
        
        left_leg_risk = 0.0
        right_leg_risk = 0.0
        
        if pd.notna(left_knee_angle):
            if is_landing:
                if left_knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    left_leg_risk += 60.0
                elif left_knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    left_leg_risk += 40.0
                else:
                    left_leg_risk += 15.0
            elif is_jump:
                if left_knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    left_leg_risk += 10.0
                elif left_knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    left_leg_risk += 5.0
                else:
                    left_leg_risk += 2.0
            else:
                if left_knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    left_leg_risk += 50.0
                elif left_knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    left_leg_risk += 25.0
                else:
                    left_leg_risk += 5.0
        
        if pd.notna(right_knee_angle):
            if is_landing:
                if right_knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    right_leg_risk += 60.0
                elif right_knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    right_leg_risk += 40.0
                else:
                    right_leg_risk += 15.0
            elif is_jump:
                if right_knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    right_leg_risk += 10.0
                elif right_knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    right_leg_risk += 5.0
                else:
                    right_leg_risk += 2.0
            else:
                if right_knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    right_leg_risk += 50.0
                elif right_knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    right_leg_risk += 25.0
                else:
                    right_leg_risk += 5.0
        
        if left_leg_risk > 0 or right_leg_risk > 0:
            score += max(left_leg_risk, right_leg_risk)
            factors += 1
        elif pd.notna(knee_angle):
            if is_landing:
                landing_knee_angle = row.get("knee_angle_during_landing", knee_angle)
                if landing_knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    score += 60.0
                elif landing_knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    score += 40.0
                else:
                    score += 15.0
            elif is_jump:
                if knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    score += 10.0
                elif knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    score += 5.0
                else:
                    score += 2.0
            else:
                if knee_angle < self.KNEE_ANGLE_HIGH_RISK_THRESHOLD:
                    score += 50.0
                elif knee_angle < self.KNEE_ANGLE_MEDIUM_RISK_THRESHOLD:
                    score += 25.0
                else:
                    score += 5.0
            factors += 1
        
        if is_landing:
            landing_force = row.get("landing_force_estimate", 0)
            if landing_force > 0.05:
                score += 20.0
            elif landing_force > 0.02:
                score += 10.0
        

        if pd.notna(row.get("knee_angular_velocity")):
            angular_vel = abs(row["knee_angular_velocity"]) * 30.0
            if angular_vel > self.ANGULAR_VELOCITY_HIGH_THRESHOLD:
                score += 30.0
            elif angular_vel > self.ANGULAR_VELOCITY_HIGH_THRESHOLD * 0.5:
                score += 20.0
            else:
                score += 5.0
            factors += 1
        

        if pd.notna(row.get("hip_angle")):
            hip_angle = row["hip_angle"]
            if hip_angle < self.HIP_ANGLE_OPTIMAL_MIN or hip_angle > self.HIP_ANGLE_OPTIMAL_MAX:
                score += 10.0
            else:
                score += 2.0
            factors += 1
        

        if pd.notna(row.get("ankle_angle")):
            ankle_angle = row["ankle_angle"]
            if ankle_angle < 60.0 or ankle_angle > 130.0:
                score += 10.0
            else:
                score += 2.0
            factors += 1
        


        if is_cutting:
            cutting_asymmetry = row.get("asymmetry_during_cutting", 0)
            if not pd.isna(cutting_asymmetry):
                if cutting_asymmetry > 20.0:
                    score += 35.0
                elif cutting_asymmetry > 10.0:
                    score += 20.0
                else:
                    score += 5.0
            

            cutting_angle = row.get("cutting_angle", 0)
            if not pd.isna(cutting_angle) and cutting_angle > 45.0:
                score += 15.0
        

        if is_jump or is_landing:
            jump_height = row.get("jump_height", 0)
            if not pd.isna(jump_height) and jump_height > 0.15:
                score += 10.0
        
        if factors > 0:
            return min(score, 100.0)
        else:
            return 50.0
    
    def _score_to_label(self, score):

        if score >= 50.0:
                return "High"
        elif score >= 25.0:
                return "Medium"
        else:
            return "Low"
