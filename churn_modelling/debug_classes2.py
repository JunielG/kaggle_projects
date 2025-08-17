import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.base import clone
from collections import Counter
import warnings

def robust_learning_curve(estimator, X, y, cv_splits=5, min_samples_per_class=2, 
                         max_train_sizes=8, scoring='accuracy', random_state=42):
    """
    Create learning curves with robust handling of imbalanced datasets
    """
    
    # Analyze dataset
    class_counts = Counter(y)
    total_samples = len(y)
    min_class_size = min(class_counts.values())
    n_classes = len(class_counts)
    
    print(f"Dataset: {total_samples} samples, {n_classes} classes")
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Smallest class: {min_class_size} samples")
    
    # Calculate safe minimum training size
    # Each fold needs at least min_samples_per_class from each class
    min_total_per_fold = min_samples_per_class * n_classes
    min_training_fraction = (min_total_per_fold * cv_splits) / total_samples
    
    # Be more conservative - multiply by 1.5 for safety
    safe_min_fraction = min(0.8, max(0.3, min_training_fraction * 1.5))
    
    print(f"Calculated minimum safe training fraction: {safe_min_fraction:.3f}")
    
    # Create training sizes
    train_sizes = np.linspace(safe_min_fraction, 1.0, max_train_sizes)
    
    # Use stratified cross-validation
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    try:
        # First attempt with standard learning_curve
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, 
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=1,  # Use single job to avoid multiprocessing issues
            error_score='raise',
            random_state=random_state
        )
        
        return train_sizes_abs, train_scores, test_scores
        
    except Exception as e:
        print(f"Standard learning_curve failed: {e}")
        print("Falling back to manual implementation...")
        
        # Manual implementation
        return manual_learning_curve(estimator, X, y, cv, train_sizes, scoring)

def manual_learning_curve(estimator, X, y, cv, train_sizes, scoring):
    """
    Manual implementation of learning curve with better error handling
    """
    from sklearn.metrics import accuracy_score
    
    train_sizes_abs = []
    train_scores_list = []
    test_scores_list = []
    
    for train_size in train_sizes:
        fold_train_scores = []
        fold_test_scores = []
        actual_train_size = int(train_size * len(y))
        
        for train_idx, test_idx in cv.split(X, y):
            # Limit training set size
            if len(train_idx) > actual_train_size:
                # Stratified sampling to maintain class balance
                train_idx_limited = stratified_sample(train_idx, y[train_idx], actual_train_size)
            else:
                train_idx_limited = train_idx
            
            X_train_fold = X[train_idx_limited]
            y_train_fold = y[train_idx_limited]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]
            
            # Check if we have all classes in training set
            train_classes = np.unique(y_train_fold)
            if len(train_classes) < len(np.unique(y)):
                print(f"Skipping fold - missing classes in training set: {train_classes}")
                continue
            
            try:
                # Clone and fit estimator
                model = clone(estimator)
                model.fit(X_train_fold, y_train_fold)
                
                # Calculate scores
                train_pred = model.predict(X_train_fold)
                test_pred = model.predict(X_test_fold)
                
                if scoring == 'accuracy':
                    train_score = accuracy_score(y_train_fold, train_pred)
                    test_score = accuracy_score(y_test_fold, test_pred)
                else:
                    # Add other scoring metrics as needed
                    train_score = accuracy_score(y_train_fold, train_pred)
                    test_score = accuracy_score(y_test_fold, test_pred)
                
                fold_train_scores.append(train_score)
                fold_test_scores.append(test_score)
                
            except Exception as e:
                print(f"Error in fold: {e}")
                continue
        
        if fold_train_scores:  # Only add if we have valid scores
            train_sizes_abs.append(actual_train_size)
            train_scores_list.append(fold_train_scores)
            test_scores_list.append(fold_test_scores)
    
    # Convert to numpy arrays with proper shape
    train_sizes_abs = np.array(train_sizes_abs)
    
    # Pad shorter lists with NaN to make rectangular array
    max_folds = max(len(scores) for scores in train_scores_list) if train_scores_list else 0
    
    train_scores = np.full((len(train_scores_list), max_folds), np.nan)
    test_scores = np.full((len(test_scores_list), max_folds), np.nan)
    
    for i, scores in enumerate(train_scores_list):
        train_scores[i, :len(scores)] = scores
    
    for i, scores in enumerate(test_scores_list):
        test_scores[i, :len(scores)] = scores
    
    return train_sizes_abs, train_scores, test_scores

def stratified_sample(indices, labels, target_size):
    """
    Stratified sampling to maintain class proportions
    """
    from sklearn.model_selection import train_test_split
    
    if len(indices) <= target_size:
        return indices
    
    # Calculate how many samples we need from each class
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    sampled_indices = []
    
    for label in unique_labels:
        label_indices = indices[labels == label]
        n_needed = int((counts[unique_labels == label][0] / total_samples) * target_size)
        n_needed = max(1, min(n_needed, len(label_indices)))  # At least 1, at most available
        
        if len(label_indices) > n_needed:
            sampled = np.random.choice(label_indices, n_needed, replace=False)
        else:
            sampled = label_indices
        
        sampled_indices.extend(sampled)
    
    return np.array(sampled_indices)

def plot_learning_curve(train_sizes, train_scores, test_scores, title="Learning Curve"):
    """
    Plot learning curve with error bars
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate means and standard deviations
    train_mean = np.nanmean(train_scores, axis=1)
    train_std = np.nanstd(train_scores, axis=1)
    test_mean = np.nanmean(test_scores, axis=1)
    test_std = np.nanstd(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Example usage:
train_sizes, train_scores, test_scores = robust_learning_curve(
    best_model, X, y, cv_splits=5, min_samples_per_class=3
)
plot_learning_curve(train_sizes, train_scores, test_scores)