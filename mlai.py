import pandas as pd
import numpy as np
import warnings
import os
# Suppress pandas fragmentation warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA PREPARATION & CLEANING
# ==========================================

def load_and_prep_data(file_path: str, variance_threshold: float = 0.01) -> pd.DataFrame:
    """Loads the Parquet file, pivots it for ML, and removes flatline metrics."""
    print(f"📥 Loading Parquet file from: {file_path}")
    df_raw = pd.read_parquet(file_path)
    
    print("🔄 Pivoting data into Time-Series Matrix...")
    # Convert from Long (DB style) to Wide (ML style)
    # index = time, columns = items, values = sensor readings
    df_wide = df_raw.pivot_table(index='clock', columns='name', values='value')
    
    # Sort by time just in case the database returned it out of order
    df_wide = df_wide.sort_index()
    
    # Convert UNIX epoch time to human-readable datetime index
    df_wide.index = pd.to_datetime(df_wide.index, unit='s')
    
    # Handle missing data: Forward-fill the last known value, then fill any remaining with 0
    df_wide = df_wide.ffill().fillna(0)
    
    # THE GREAT PURGE: Drop columns that do not change (flatlines)
    print(f"📊 Original matrix shape: {df_wide.shape} (Rows, Columns)")
    variances = df_wide.var()
    active_columns = variances[variances > variance_threshold].index
    
    df_active = df_wide[active_columns]
    print(f"🧹 Filtered matrix shape: {df_active.shape} (Active items only)")
    
    return df_active

# ==========================================
# 2. MACHINE LEARNING ALGORITHMS
# ==========================================
def find_concurrent_relationships(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Finds pairs of items that consistently spike or drop at the exact same time."""
    print("\n🧮 Calculating Concurrent (Simultaneous) Relationships...")
    
    # Calculate Pearson Correlation
    corr_matrix = df.corr(method='pearson')
    
    # Flatten the matrix into pairs (this creates a Series with a MultiIndex)
    pairs = corr_matrix.unstack().dropna()
    
    # Filter for strong relationships, ignoring self-correlation (1.0)
    strong_pairs = pairs[(abs(pairs) >= threshold) & (pairs < 1.0)]
    
    # Guard against empty results (if nothing meets the threshold)
    if strong_pairs.empty:
         return pd.DataFrame(columns=['Item A', 'Item B', 'Correlation Score'])
         
    # FIX: Rename the MultiIndex levels BEFORE resetting the index
    # This prevents the "cannot insert itemid, already exists" error
    strong_pairs.index.names = ['Item A', 'Item B']
    
    # Now reset the index safely
    results_df = strong_pairs.reset_index()
    
    # Rename the columns explicitly (the correlation value column usually defaults to 0)
    results_df.columns = ['Item A', 'Item B', 'Correlation Score']
    
    # Drop duplicates (Since A->B is the same as B->A)
    # We create a temporary key that sorts the pair, drop duplicates based on that key, then remove it
    results_df['Pair_Key'] = results_df.apply(lambda row: tuple(sorted([row['Item A'], row['Item B']])), axis=1)
    results_df = results_df.drop_duplicates(subset=['Pair_Key']).drop(columns=['Pair_Key'])
    
    # Sort by highest correlation
    results_df = results_df.sort_values(by='Correlation Score', ascending=False)
    
    return results_df

def find_anticipating_relationships(df: pd.DataFrame, target_item: int, max_lags: int = 5) -> pd.DataFrame:
    """
    Cross-Correlation: Finds items that spike BEFORE the target item.
    max_lags determines how many time-steps to look backward.
    """
    print(f"\n🔮 Looking for Leading Indicators for Target name: {target_item}...")
    
    if target_item not in df.columns:
        print(f"⚠️ Target Item {target_item} was filtered out or doesn't exist.")
        return pd.DataFrame()

    results = []
    target_series = df[target_item]

    for col in df.columns:
        if col == target_item:
            continue
            
        best_corr = 0
        best_lag = 0
        
        # Shift the potential leader's timeline FORWARD to see if it matches the target
        for lag in range(1, max_lags + 1):
            shifted_series = df[col].shift(lag)
            corr = target_series.corr(shifted_series)
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
                
        # If the highest correlation across the shifts is strong (> 0.7)
        if abs(best_corr) > 0.7:
            results.append({
                'Leading Item': col,
                'Correlation': best_corr,
                'Time Steps Ahead': best_lag
            })
            
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Correlation', ascending=False)
        
    return results_df

# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    hosts_path = "./output_data"
    dir_list = os.listdir(hosts_path)
    for hp in dir_list:
        tag_list = os.listdir(f"{hosts_path}/{hp}")
        for tp in tag_list:
            parquet_name = os.listdir(f"{hosts_path}/{hp}/{tp}")[0]
            parquet_dir = f"{hosts_path}/{hp}/{tp}/{parquet_name}"
            df_ml = load_and_prep_data(parquet_dir, variance_threshold=0.05)
                        # 1. Load and prep data (Adjust variance_threshold if too many/too few items are dropped)
            if not df_ml.empty:
                # 2. Find Concurrent Spikes
                concurrent_df = find_concurrent_relationships(df_ml, threshold=0.85)
                print("\n--- TOP 5 CONCURRENT RELATIONSHIPS ---")
                if not concurrent_df.empty:
                    print(concurrent_df.head(5).to_string(index=False))
                else:
                    print("No concurrent relationships found above the threshold.")
                    
                # 3. Find Anticipating Spikes
                # Pick the very first active item as a test target. 
                # In the future, replace this with a specific ItemID you care about (e.g., test_target = 12345)
                test_target = df_ml.columns[0] 
                
                anticipating_df = find_anticipating_relationships(df_ml, target_item=test_target, max_lags=10)
                
                print(f"\n--- ITEMS THAT SPIKE BEFORE {test_target} ---")
                if not anticipating_df.empty:
                    print(anticipating_df.head(5).to_string(index=False))
                else:
                    print(f"No strong leading indicators found for {test_target}.")
            else:
                print("Dataset is empty after variance filtering. Try lowering the variance_threshold.")
    
    # # 1. Load and prep data (Adjust variance_threshold if too many/too few items are dropped)
    # df_ml = load_and_prep_data(FILE_PATH, variance_threshold=0.05)
    
    # if not df_ml.empty:
    #     # 2. Find Concurrent Spikes
    #     concurrent_df = find_concurrent_relationships(df_ml, threshold=0.85)
    #     print("\n--- TOP 5 CONCURRENT RELATIONSHIPS ---")
    #     if not concurrent_df.empty:
    #         print(concurrent_df.head(5).to_string(index=False))
    #     else:
    #         print("No concurrent relationships found above the threshold.")
            
    #     # 3. Find Anticipating Spikes
    #     # Pick the very first active item as a test target. 
    #     # In the future, replace this with a specific ItemID you care about (e.g., test_target = 12345)
    #     test_target = df_ml.columns[0] 
        
    #     anticipating_df = find_anticipating_relationships(df_ml, target_item=test_target, max_lags=10)
        
    #     print(f"\n--- ITEMS THAT SPIKE BEFORE {test_target} ---")
    #     if not anticipating_df.empty:
    #         print(anticipating_df.head(5).to_string(index=False))
    #     else:
    #         print(f"No strong leading indicators found for {test_target}.")
    # else:
    #     print("Dataset is empty after variance filtering. Try lowering the variance_threshold.")