
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.ml_models.random_forest_model import RandomForestModel
from src.ml_models.svm_model import SVMModel
from src.ml_models.xgboost_model import XGBoostModel


def train_all_models(data_path="data/features/labeled_features_combined.csv", 
                     test_size=0.2, 
                     random_state=42,
                     output_dir="models"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Feature data not found: {data_path}\n"
            "Please run pose extraction and feature extraction first."
        )
    
    df = pd.read_csv(data_path)
    
    required_cols = ["knee_angle", "knee_angular_velocity", "risk_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    feature_cols = ["knee_angle", "knee_angular_velocity"]
    

    if "hip_angle" in df.columns:
        feature_cols.append("hip_angle")
    if "ankle_angle" in df.columns:
        feature_cols.append("ankle_angle")
    if "is_jump" in df.columns:
        feature_cols.append("is_jump")
    if "is_landing" in df.columns:
        feature_cols.append("is_landing")
    if "is_cutting" in df.columns:
        feature_cols.append("is_cutting")
    if "knee_angle_during_landing" in df.columns:
        feature_cols.append("knee_angle_during_landing")
    if "asymmetry_during_cutting" in df.columns:
        feature_cols.append("asymmetry_during_cutting")
    if "landing_force_estimate" in df.columns:
        feature_cols.append("landing_force_estimate")
    if "jump_height" in df.columns:
        feature_cols.append("jump_height")
    if "cutting_angle" in df.columns:
        feature_cols.append("cutting_angle")
    if "hip_velocity_y" in df.columns:
        feature_cols.append("hip_velocity_y")
    if "ankle_velocity_y" in df.columns:
        feature_cols.append("ankle_velocity_y")
    if "ankle_acceleration_y" in df.columns:
        feature_cols.append("ankle_acceleration_y")
    
    X = df[feature_cols].copy()
    y = df["risk_label"].copy()
    
    X = X.fillna(X.mean())
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    print(f"Label mapping: {label_mapping}")
    
    is_binary = len(le.classes_) == 2
    print(f"Classification type: {'Binary' if is_binary else 'Multi-class'}")
    
    unique, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = dict(zip([le.classes_[i] for i in unique], counts))
    print(f"Class distribution: {class_distribution}")
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed ({str(e)}). Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state
        )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {feature_cols}")
    
    models = {
        "random_forest": RandomForestModel(),
        "svm": SVMModel(),
        "xgboost": XGBoostModel()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}...")
        print(f"{'='*60}")
        
        try:
            model.train(X_train, y_train, class_weights=class_weight_dict)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            if is_binary:
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            else:
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "y_test": y_test,
                "y_pred": y_pred
            }
            
            print(f"\nTest Set Performance:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            print(f"\nClassification Report:")
            try:
                print(classification_report(
                    y_test, y_pred,
                    target_names=[label_mapping[i] for i in range(len(label_mapping))],
                    zero_division=0
                ))
            except ValueError as e:
                print(f"Note: Some classes may be missing in test set: {str(e)}")
                unique_test = np.unique(y_test)
                unique_pred = np.unique(y_pred)
                print(f"Classes in test set: {[label_mapping[i] for i in unique_test]}")
                print(f"Classes in predictions: {[label_mapping[i] for i in unique_pred]}")
            
            model_path = os.path.join(output_dir, f"{model_name}.pkl")
            model.save(model_path)
            print(f"\n✅ Model saved to: {model_path}")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for model_name, result in results.items():
        if "error" not in result:
            print(f"{model_name:<20} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
                  f"{result['recall']:<12.4f} {result['f1_score']:<12.4f}")
        else:
            print(f"{model_name:<20} ERROR: {result['error']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train machine learning models for basketball injury risk analysis"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/features/labeled_features_combined.csv",
        help="Path to labeled feature CSV file"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)"
    )
    
    args = parser.parse_args()
    
    results = train_all_models(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
    
    print("\n✅ Training complete!")
