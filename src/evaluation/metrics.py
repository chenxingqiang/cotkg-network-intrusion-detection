from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(y_true, y_pred, class_names):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compute classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True)

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Print classification report
    for cls in class_names:
        print(f"{cls}:")
        print(f"  Precision: {report[cls]['precision']:.2f}")
        print(f"  Recall: {report[cls]['recall']:.2f}")
        print(f"  F1-score: {report[cls]['f1-score']:.2f}")
        print()

    print(f"Accuracy: {report['accuracy']:.2f}")

    return cm, report
