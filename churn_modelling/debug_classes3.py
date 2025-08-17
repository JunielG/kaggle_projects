from sklearn.model_selection import learning_curve, StratifiedKFold
import numpy as np
from collections import Counter

# Check your data first
print("Class distribution:", Counter(y))
print("Dataset size:", len(y))

# Use more conservative training sizes and stratified CV
min_class_size = min(Counter(y).values())
total_samples = len(y)

# Calculate a safe minimum training size
# Need at least 2 samples per class per fold
safe_min_fraction = max(0.4, (10 * len(np.unique(y))) / total_samples)

train_sizes = np.linspace(safe_min_fraction, 1.0, 6)  # Fewer points, larger minimum

stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Fewer folds

try:
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y, 
        cv=stratified_cv, 
        n_jobs=1,  # Single job to avoid multiprocessing issues
        train_sizes=train_sizes,
        scoring='accuracy', 
        error_score='raise',
        random_state=42
    )
    print("Learning curve completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("Your dataset might be too small or severely imbalanced.")