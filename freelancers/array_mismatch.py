# Method 1: Average feature importance across all estimators
def get_feature_importance_bagging2(bagging_model, feature_names):
    """Extract feature importance from BaggingClassifier"""
    # Get feature importance from each estimator
    importances = []
    for estimator in bagging_model.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            importances.append(estimator.feature_importances_)
    
    # Average the importances
    mean_importance = np.mean(importances, axis=0)
    std_importance = np.std(importances, axis=0)
    
    return mean_importance, std_importance

# Get feature importance
mean_importance, std_importance = get_feature_importance_bagging2(best_bag_clf, feature_names)

try:
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importance,
        'Std': std_importance
    }).sort_values('Importance', ascending=False)
    print("Feature Importance (Bagging Classifier):")
    print(importance_df)
    
except ValueError as e:
    print(f"Length mismatch: {e}")
    print(f"Lengths - Features: {len(feature_names)}, Importance: {len(mean_importance)}, Std: {len(std_importance)}")
    print('-' *127)
    # Handle the mismatch appropriately