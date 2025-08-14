from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

def metrics_binary(y_true, y_proba):
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "auroc": float(roc_auc_score(y_true, y_proba)),
        "logloss": float(log_loss(y_true, y_proba, labels=[0,1])),
    }
