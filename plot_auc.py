from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_auc(model, test_dataset, tasks):
    # Maybe we should first apply a transformer to the test dataset and then predict 
    # Originally predictions stores GraphData objects
    # But somehow the model manages
    predictions = model.predict(test_dataset)

    y_true = test_dataset.y
    w = test_dataset.w

    # Plot ROC curves
    print("\nGenerating ROC curves...")

    # Plot ROC curve for each task
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for task_idx, task_name in enumerate(tasks):
        mask = w[:, task_idx] > 0
        fpr, tpr, _ = roc_curve(y_true[mask, task_idx], predictions[mask, task_idx, 1])
        roc_auc = auc(fpr, tpr)
        
        axes[task_idx].plot(fpr, tpr, color='darkorange', lw=2, 
                            label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[task_idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[task_idx].set_xlim([0.0, 1.0])
        axes[task_idx].set_ylim([0.0, 1.05])
        axes[task_idx].set_xlabel('False Positive Rate')
        axes[task_idx].set_ylabel('True Positive Rate')
        axes[task_idx].set_title(f'{task_name}')
        axes[task_idx].legend(loc="lower right")
        axes[task_idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('ensemble_roc_curves.png', dpi=150)
    print("ROC curves saved as 'ensemble_roc_curves.png'")
