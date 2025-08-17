import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Visualize feature importance for BaggingClassifier
def plot_bagging_feature_importance(bagging_clf, X_train, top_n=10):
    """
    Plot feature importance for BaggingClassifier by averaging across all estimators
    
    Parameters:
    - bagging_clf: trained BaggingClassifier
    - X_train: training features (pandas DataFrame or numpy array with feature names)
    - top_n: number of top features to display
    """
    
    # Check if the base estimator supports feature importance
    if not hasattr(bagging_clf.estimators_[0], 'feature_importances_'):
        print("Base estimators don't support feature importances")
        return
    
    # Get feature names
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # Calculate average feature importance across all estimators
    n_features = len(feature_names)
    avg_importance = np.zeros(n_features)
    
    for estimator in bagging_clf.estimators_:
        avg_importance += estimator.feature_importances_
    
    avg_importance /= len(bagging_clf.estimators_)
    
    # Get top N features
    indices = np.argsort(avg_importance)[::-1][:top_n]
    top_importances = avg_importance[indices]
    top_feature_names = [feature_names[i] for i in indices]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(top_n), top_importances, color='skyblue', alpha=0.8)
    
    plt.title(f'Top {top_n} Feature Importances (BaggingClassifier)', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Average Importance', fontsize=12)
    
    # Set x-axis labels
    plt.xticks(range(top_n), top_feature_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print feature importance values
    print(f"\nTop {top_n} Feature Importances:")
    print("-" * 40)
    for i, (name, importance) in enumerate(zip(top_feature_names, top_importances)):
        print(f"{i+1:2d}. {name:20s}: {importance:.4f}")

# Alternative: Get feature importance statistics across estimators
def get_feature_importance_stats(bagging_clf, X_train):
    """
    Get statistical summary of feature importances across all estimators
    """
    if not hasattr(bagging_clf.estimators_[0], 'feature_importances_'):
        print("Base estimators don't support feature importances")
        return None
    
    # Get feature names
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # Collect importance from all estimators
    all_importances = []
    for estimator in bagging_clf.estimators_:
        all_importances.append(estimator.feature_importances_)
    
    all_importances = np.array(all_importances)
    
    # Calculate statistics
    stats = {
        'mean': np.mean(all_importances, axis=0),
        'std': np.std(all_importances, axis=0),
        'min': np.min(all_importances, axis=0),
        'max': np.max(all_importances, axis=0)
    }
    
    return feature_names, stats

# Usage example:
# Assuming you have a trained BaggingClassifier called 'best_bag_clf'
# plot_bagging_feature_importance(best_bag_clf, X_train, top_n=10)

# Or get detailed statistics:
# feature_names, importance_stats = get_feature_importance_stats(best_bag_clf, X_train)

# Example of creating and using a BaggingClassifier:
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train BaggingClassifier
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)
bagging_clf.fit(X_train, y_train)

# Plot feature importance
plot_bagging_feature_importance(bagging_clf, X_train, top_n=10)
"""