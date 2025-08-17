from sklearn.model_selection import learning_curve, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

# Check class distribution
print("Class distribution:", np.bincount(y))
min_class_count = np.min(np.bincount(y))
print(f"Minimum class has {min_class_count} samples")

# Use stratified CV
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Adjust train_sizes based on minimum class count
# Ensure each fold has at least a few samples of each class
min_samples_per_fold = max(10, min_class_count // 5)  # At least 10 or 1/5 of min class
min_train_size = min_samples_per_fold / len(y)
min_train_size = max(min_train_size, 0.1)  # At least 10% of data

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, 
    cv=stratified_cv, 
    n_jobs=-1,
    train_sizes=np.linspace(min_train_size, 1.0, 10),
    scoring='accuracy',
    error_score=np.nan  # Handle failed fits gracefully
)

# Remove any NaN scores
train_scores_mean = np.nanmean(train_scores, axis=1)
train_scores_std = np.nanstd(train_scores, axis=1)
test_scores_mean = np.nanmean(test_scores, axis=1)
test_scores_std = np.nanstd(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.plot(train_sizes, test_scores_mean, 'o-', color='red', label='Cross-validation score')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()