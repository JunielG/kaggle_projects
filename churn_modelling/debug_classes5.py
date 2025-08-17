# First, let's diagnose the problem
print("Class distribution:")
print(y.value_counts())
print(f"Total samples: {len(y)}")
print(f"Number of classes: {y.nunique()}")

# Solution 1: Use Stratified Cross-Validation (Recommended)
from sklearn.model_selection import StratifiedKFold

print("\n--- Model With Hyperparameter Tuning (Stratified CV) ---")
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__max_iter': [100, 500, 1000],
    'classifier__class_weight': [None, 'balanced']
}

# Use stratified k-fold to ensure each fold has both classes
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=stratified_cv,  # Use stratified CV
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Check if we have enough samples for stratification
min_class_count = y.value_counts().min()
if min_class_count < 5:
    print(f"Warning: Minimum class has only {min_class_count} samples.")
    print("Reducing number of CV folds...")
    stratified_cv = StratifiedKFold(n_splits=min(3, min_class_count), shuffle=True, random_state=42)
    grid_search.cv = stratified_cv

try:
    grid_search.fit(X_train, y_train)
    print("Grid search completed successfully!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
except ValueError as e:
    print(f"Error occurred: {e}")
    print("Trying alternative solutions...")
    
    # Solution 2: Reduce parameter grid and CV folds
    print("\n--- Alternative: Reduced Parameter Grid ---")
    param_grid_small = {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],  # Only L2 regularization
        'classifier__solver': ['liblinear'],  # Only one solver
        'classifier__max_iter': [1000],
        'classifier__class_weight': ['balanced']  # Focus on balanced weights
    }
    
    # Use fewer folds if dataset is very small
    n_folds = min(3, min_class_count)
    small_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    grid_search_small = GridSearchCV(
        pipeline,
        param_grid=param_grid_small,
        cv=small_cv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    try:
        grid_search_small.fit(X_train, y_train)
        print("Reduced grid search completed!")
        print(f"Best parameters: {grid_search_small.best_params_}")
        print(f"Best cross-validation score: {grid_search_small.best_score_:.4f}")
        
        # Use the best model
        best_model = grid_search_small.best_estimator_
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
    except ValueError as e2:
        print(f"Still getting error: {e2}")
        
        # Solution 3: Manual hyperparameter testing
        print("\n--- Manual Hyperparameter Testing ---")
        
        # Test a few configurations manually
        configs = [
            {'C': 1, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000, 'class_weight': 'balanced'},
            {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000, 'class_weight': 'balanced'},
            {'C': 10, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000, 'class_weight': 'balanced'}
        ]
        
        best_score = 0
        best_config = None
        
        for config in configs:
            # Create pipeline with specific parameters
            manual_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(**config, random_state=42))
            ])
            
            # Use cross-validation manually
            try:
                scores = cross_val_score(manual_pipeline, X_train, y_train, 
                                       cv=small_cv, scoring='accuracy')
                mean_score = scores.mean()
                print(f"Config {config}: CV Score = {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config
                    
            except ValueError as e3:
                print(f"Config {config} failed: {e3}")
                continue
        
        if best_config:
            print(f"\nBest manual configuration: {best_config}")
            print(f"Best CV score: {best_score:.4f}")
            
            # Train final model
            final_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(**best_config, random_state=42))
            ])
            final_pipeline.fit(X_train, y_train)
            y_pred = final_pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Test accuracy: {test_accuracy:.4f}")

# Additional diagnostic information
print(f"\n--- Dataset Analysis ---")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print("Training set class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print("Test set class distribution:")
print(pd.Series(y_test).value_counts().sort_index())