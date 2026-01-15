import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import ast
import numpy as np

def run_script_and_get_metrics(script_path):
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    output = result.stdout

    try:
    
        accuracy = float(re.search(r'Accuracy:\s+([0-9.]+)', output).group(1))
        precision = float(re.search(r'Precision:\s+([0-9.]+)', output).group(1))
        recall = float(re.search(r'Recall:\s+([0-9.]+)', output).group(1))
        f1 = float(re.search(r'F1-Score:\s+([0-9.]+)', output).group(1))

        
        try:
            auc_value = float(re.search(r'AUC:\s+([0-9.]+)', output).group(1))
        except:
            auc_value = 0

        try:
            y_true = ast.literal_eval(re.search(r'True Labels:\s+(\[.*?\])', output).group(1))
            y_pred = ast.literal_eval(re.search(r'Predicted Probs:\s+(\[.*?\])', output).group(1))
        except:
            y_true, y_pred = [], []

        
        if y_true and y_pred:
            y_true_bin = label_binarize(y_true, classes=list(set(y_true)))
            if y_true_bin.shape[1] == 1:  # binary
                y_true_bin = y_true_bin.ravel()
            else:  
                y_true_bin = y_true_bin[:, 0]
            y_pred_bin = np.array([1 if p == y_true[i] else 0 for i, p in enumerate(y_pred)])
            auc_value = auc(*roc_curve(y_true_bin, y_pred_bin)[:2])
        else:
            y_true_bin = []
            y_pred_bin = []

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc_value,
            "y_true": y_true_bin,
            "y_scores": y_pred_bin
        }

    except Exception as e:
        print(f"❌ Error extracting metrics from {script_path}:\n{output}")
        return {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1-Score": 0,
            "AUC": 0,
            "y_true": [],
            "y_scores": []
        }


yolov8_metrics = run_script_and_get_metrics("yolo_motion_tracker.py")
yolov5_metrics = run_script_and_get_metrics("yolov5.py")


df = pd.DataFrame([
    {k: yolov8_metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]},
    {k: yolov5_metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]}
], index=["YOLOv8", "YOLOv5"])


print("\n📋 YOLOv5 vs YOLOv8 - Detection Metric Comparison:\n")
print(tabulate(df, headers='keys', tablefmt='fancy_grid'))


df.drop(columns=["AUC"]).plot(kind='bar', figsize=(10, 6), rot=0, colormap='viridis')
plt.title("YOLOv5 vs YOLOv8 - Performance Comparison")
plt.ylabel("Score (0 to 1)")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
for model_name, metrics in zip(["YOLOv8", "YOLOv5"], [yolov8_metrics, yolov5_metrics]):
    if metrics["y_true"] and metrics["y_scores"]:
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["y_scores"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve - YOLOv5 vs YOLOv8")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
