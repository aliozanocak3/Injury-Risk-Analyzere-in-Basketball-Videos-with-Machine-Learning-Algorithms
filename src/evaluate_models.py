
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.ml_models.random_forest_model import RandomForestModel
from src.ml_models.svm_model import SVMModel
from src.ml_models.xgboost_model import XGBoostModel


def load_data(data_path="data/features/labeled_features.csv"):
    
    df = pd.read_csv(data_path)
    
    feature_cols = ["knee_angle", "knee_angular_velocity"]
    if "hip_angle" in df.columns:
        feature_cols.append("hip_angle")
    if "ankle_angle" in df.columns:
        feature_cols.append("ankle_angle")
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df["risk_label"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, feature_cols, label_mapping


def evaluate_model(model, model_name, X_test, y_test, label_mapping, output_dir="evaluation_results"):
   
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=[label_mapping[i] for i in range(len(label_mapping))]
        )
    }
    
    plot_confusion_matrix(cm, label_mapping, model_name, output_dir)
    
    return results


def plot_confusion_matrix(cm, label_mapping, model_name, output_dir):
   
    labels = [label_mapping[i] for i in range(len(label_mapping))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {output_path}")


def plot_feature_importance(model, model_name, feature_names, output_dir):
    
    try:
        if hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
        elif hasattr(model.model, 'coef_'):
            importances = np.abs(model.model.coef_[0])
        else:
            print(f"  Feature importance not available for {model_name}")
            return
        
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_names)), importances[indices])
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.ylabel('Importance', fontsize=12)
        plt.xlabel('Feature', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Feature importance plot saved: {output_path}")
        
    except Exception as e:
        print(f"  Could not generate feature importance for {model_name}: {str(e)}")


def compare_models(results_dict, output_dir):
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, results in results_dict.items():
        if "error" not in results:
            model_names.append(model_name.replace("_", " ").title())
            accuracies.append(results["accuracy"])
            precisions.append(results["precision"])
            recalls.append(results["recall"])
            f1_scores.append(results["f1_score"])
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved: {output_path}")
    
    summary_data = {
        "Model": model_names,
        "Accuracy": [f"{a:.4f}" for a in accuracies],
        "Precision": [f"{p:.4f}" for p in precisions],
        "Recall": [f"{r:.4f}" for r in recalls],
        "F1-Score": [f"{f:.4f}" for f in f1_scores]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved: {summary_path}")
    print("\n" + summary_df.to_string(index=False))


def main():
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_names, label_mapping = load_data()
    print(f"Test set size: {len(X_test)} samples")
    print(f"Features: {feature_names}")
    
    models = {
        "random_forest": RandomForestModel(),
        "svm": SVMModel(),
        "xgboost": XGBoostModel()
    }
    
    results_dict = {}
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        model_path = f"models/{model_name}.pkl"
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {model_path}")
            print("   Please train models first using: python src/train_model.py")
            results_dict[model_name] = {"error": "Model file not found"}
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()}...")
        print(f"{'='*60}")
        
        try:
            model.load(model_path)
            
            results = evaluate_model(model, model_name, X_test, y_test, label_mapping, output_dir)
            results_dict[model_name] = results
            
            print(f"\nMetrics:")
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-Score:  {results['f1_score']:.4f}")
            
            plot_feature_importance(model, model_name, feature_names, output_dir)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            results_dict[model_name] = {"error": str(e)}
    
    if len([r for r in results_dict.values() if "error" not in r]) > 0:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        compare_models(results_dict, output_dir)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

