import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def analyze_dataset_for_learning_curve(X, y):
    """
    Comprehensive analysis of dataset to identify learning curve issues
    """
    print("=== DATASET ANALYSIS ===")
    
    # Basic info
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Class distribution
    class_counts = Counter(y)
    print(f"\nClass distribution: {dict(class_counts)}")
    
    unique_classes = np.unique(y)
    print(f"Unique classes: {unique_classes}")
    print(f"Number of classes: {len(unique_classes)}")
    
    # Check for class imbalance
    total_samples = len(y)
    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")
    
    # Calculate minimum viable training sizes
    min_class_size = min(class_counts.values())
    max_class_size = max(class_counts.values())
    print(f"\nSmallest class has {min_class_size} samples")
    print(f"Largest class has {min_class_size} samples\n")
    
    # Simulate cross-validation splits to see what happens
    print("\n=== CROSS-VALIDATION SIMULATION ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_y_fold = y[train_idx]
        test_y_fold = y[test_idx]
        
        train_classes = np.unique(train_y_fold)
        test_classes = np.unique(test_y_fold)
        
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train classes: {train_classes} (counts: {Counter(train_y_fold)})")
        print(f"  Test classes: {test_classes} (counts: {Counter(test_y_fold)})")
    
    # Calculate safe training sizes
    print("\n=== RECOMMENDED TRAINING SIZES ===")
    
    # For each training size, check if it would have enough samples per class
    test_sizes = np.linspace(0.1, 1.0, 10)
    safe_sizes = []
    
    for size in test_sizes:
        min_samples_in_fold = int(size * total_samples / 5)  # Assuming 5-fold CV
        
        # Check if smallest class would have at least 1 sample in training
        min_class_samples_in_fold = int(min_class_size * size / 5)
        
        if min_class_samples_in_fold >= 1:
            safe_sizes.append(size)
            status = "✓ SAFE"
        else:
            status = "✗ RISKY"
        
        print(f"Training size {size:.1f}: ~{min_samples_in_fold} samples per fold, "
              f"~{min_class_samples_in_fold} from smallest class - {status}")
    
    if safe_sizes:
        recommended_min = min(safe_sizes)
        print(f"\nRecommended minimum training size: {recommended_min:.2f}")
        return np.linspace(recommended_min, 1.0, min(8, len(safe_sizes)))
    else:
        print("\nWARNING: Dataset may be too small or imbalanced for reliable learning curves")
        return np.linspace(0.5, 1.0, 5)  # Very conservative approach

# Example usage:
safe_train_sizes = analyze_dataset_for_learning_curve(X, y)
safe_train_sizes