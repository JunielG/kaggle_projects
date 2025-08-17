# Final model evaluation
final_model = fine_search.best_estimator_
y_pred = final_model.predict(X_test)

# Store results for the best model
best_model_results = {
    'best_params': fine_search.best_params_,
    'cv_score': -fine_search.best_score_,  # Assuming GridSearchCV used neg_mean_squared_error
    'test_mse': mean_squared_error(y_test, y_pred),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'test_mae': mean_absolute_error(y_test, y_pred),
    'test_r2': r2_score(y_test, y_pred),
    'model': final_model
}

print("=== Best Model Results ===")
print(f"Best Parameters: {best_model_results['best_params']}")
print(f"CV Score: {best_model_results['cv_score']:.4f}")
print(f"Test MSE: {best_model_results['test_mse']:.4f}")
print(f"Test RMSE: {best_model_results['test_rmse']:.4f}")
print(f"Test MAE: {best_model_results['test_mae']:.4f}")
print(f"Test R²: {best_model_results['test_r2']:.4f}")

# If you have multiple models to compare (assuming 'models' dict exists)
# Extract MSE, MAE, and R² scores for comparison
if 'models' in locals():
    comparison_results = {}
    for name, (model, mse_scores, mae_scores, r2_scores) in models.items():
        comparison_results[name] = {
            'MSE': np.mean(mse_scores),
            'MAE': np.mean(mae_scores), 
            'R²': np.mean(r2_scores)
        }
    
    results_df = pd.DataFrame(comparison_results).T
    
    # Sort by MSE (lower is better)
    results_df = results_df.sort_values('MSE')
    
    print("\n=== Model Comparison ===")
    print(results_df)
    print("=" * 50)