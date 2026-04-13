
import os
import argparse
from src.extrract_pose import PoseExtractor
from src.calculate_features import FeatureExtractor
from src.risk_labeling import RiskLabeler
from src.train_model import train_all_models


def run_pipeline(video_path, output_dir="data", model_output_dir="models"):
    
    print("="*60)
    print("BASKETBALL INJURY RISK ANALYSIS PIPELINE")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("\n[Step 1/4] Extracting pose landmarks...")
    landmarks_path = os.path.join(output_dir, "features", "landmarks.csv")
    pose_extractor = PoseExtractor()
    pose_stats = pose_extractor.extract(video_path, landmarks_path)
    print(f"✅ Pose extraction complete: {pose_stats['frames_with_pose']} frames with pose detected")
    
    print("\n[Step 2/4] Calculating biomechanical features...")
    features_path = os.path.join(output_dir, "features", "features.csv")
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract(landmarks_path, features_path)
    print(f"✅ Feature extraction complete: {len(features_df)} frames processed")
    
    print("\n[Step 3/4] Generating risk labels (binary classification)...")
    labeled_features_path = os.path.join(output_dir, "features", "labeled_features.csv")
    risk_labeler = RiskLabeler()
    labeled_df = risk_labeler.label(features_path, labeled_features_path, binary_classification=True)
    print(f"✅ Risk labeling complete:")
    print(f"   Risk distribution:")
    risk_counts = labeled_df["risk_label"].value_counts()
    for risk, count in risk_counts.items():
        print(f"     {risk}: {count} frames ({count/len(labeled_df)*100:.1f}%)")
    
   
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - Landmarks: {landmarks_path}")
    print(f"  - Features: {features_path}")
    print(f"  - Labeled features: {labeled_features_path}")
    print(f"  - Trained models: {model_output_dir}/")
    
    return {
        "landmarks_path": landmarks_path,
        "features_path": features_path,
        "labeled_features_path": labeled_features_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete pipeline for basketball injury risk analysis"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input basketball video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory for intermediate outputs (default: data)"
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        exit(1)
    
    run_pipeline(args.video_path, args.output_dir, args.model_output_dir)

