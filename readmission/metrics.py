"""Classification metrics (same set as the original notebook)."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        'roc_auc': float(roc_auc_score(y_true, y_prob)) if both else None,
        'pr_auc': float(average_precision_score(y_true, y_prob)) if both else None,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) else 0.0,
        'brier': float(brier_score_loss(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'n': int(len(y_true)), 'positives': int(y_true.sum()),
    }
