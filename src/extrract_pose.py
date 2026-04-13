

import cv2
import mediapipe as mp
import csv
import os
import numpy as np


class PoseExtractor:
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract(self, video_path, output_csv):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_id = 0
        frames_with_pose = 0
        frames_without_pose = 0
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "landmark", "x", "y", "z", "visibility"])
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    frames_with_pose += 1
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        writer.writerow([
                            frame_id,
                            i,
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility
                        ])
                else:
                    frames_without_pose += 1
                
                frame_id += 1
                
                if frame_id % 100 == 0:
                    print(f"  Processed {frame_id}/{total_frames} frames...")
        
        cap.release()
        
        stats = {
            "total_frames": frame_id,
            "frames_with_pose": frames_with_pose,
            "frames_without_pose": frames_without_pose,
            "pose_detection_rate": frames_with_pose / frame_id if frame_id > 0 else 0.0,
            "video_fps": fps,
            "video_resolution": (width, height)
        }
        
        print(f"\nPose extraction complete:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Frames with pose: {stats['frames_with_pose']} ({stats['pose_detection_rate']*100:.1f}%)")
        print(f"  Frames without pose: {stats['frames_without_pose']}")
        
        return stats
