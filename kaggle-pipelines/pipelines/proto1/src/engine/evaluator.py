import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.logging import ConsoleLogger

def calculate_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the competition evaluation metric:
    Macro-averaged ROC-AUC, skipping classes that have no true positive labels
    in the validation split.
    
    y_true: numpy array of shape [N, 234]
    y_pred: numpy array of shape [N, 234] (probabilities after sigmoid)
    """
    auc_list = []
    
    # Iterate over all 234 classes
    for c in range(y_true.shape[1]):
        true_col = y_true[:, c]
        pred_col = y_pred[:, c]
        
        # Binary target check (convert 0.5 background label to 0 for strict evaluation, if any)
        binary_true = (true_col >= 0.5).astype(int)
        
        # Check if the class has both positive and negative samples in this validation split
        if np.sum(binary_true) > 0 and np.sum(binary_true) < len(binary_true):
            try:
                score = roc_auc_score(binary_true, pred_col)
                auc_list.append(score)
            except Exception as e:
                pass  # Skip if there's any singular numerical edge case
                
    if len(auc_list) == 0:
        return 0.5  # Random baseline fallback
        
    macro_auc = float(np.mean(auc_list))
    return macro_auc
