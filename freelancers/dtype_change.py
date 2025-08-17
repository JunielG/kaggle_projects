def dtype_change(df):
    """
    Convert specified columns to integer type if they exist in the DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame to modify
    
    Returns:
        pandas.DataFrame: DataFrame with converted column types
    """
    # Define columns that should be converted to int
    int_columns = ['age', 'years_of_experience', 'hourly_rate', 'rating', 'is_active', 'client_satisfaction']
    
    # Get actual columns that exist in the DataFrame and should be converted
    columns_to_convert = [col for col in int_columns if col in clean_df.columns]
    
    if not columns_to_convert:
        print("No matching columns found for conversion")
        return df
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = clean_df.copy()
    
    # Convert columns to int, handling potential errors
    for col in columns_to_convert:
        try:
            # Handle missing values and convert to int
            df_copy[col] = df_copy[col].fillna(0).astype('int32')
            print(f"Successfully converted '{col}' to int")
        except (ValueError, TypeError) as e:
            print(f"Could not convert '{col}' to int: {e}")
            # Optionally, try converting to float first, then int
            try:
                df_copy[col] = df_copy[col].astype('float').fillna(0).astype('int32')
                print(f"Successfully converted '{col}' to int via float")
            except:
                print(f"Failed to convert '{col}' - keeping original dtype")
    
    return df_copy

# Usage examples:
clean_df = dtype_change(clean_df)
# or
# clean_df = dtype_change_robust(clean_df)