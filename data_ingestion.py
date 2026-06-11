# src/data_ingestion.py
import pandas as pd
import os
from config import DATA_DIR, RESULTS_FILE, ELO_RATINGS_FILE, WORLD_CUP_FILE, MIN_DATE, MAX_DATE, DATE_COL

def load_dataframes():
    """
    Loads the three required CSV files into pandas DataFrames.
    """
    results_path = os.path.join(DATA_DIR, RESULTS_FILE)
    elo_path = os.path.join(DATA_DIR, ELO_RATINGS_FILE)
    world_cup_path = os.path.join(DATA_DIR, WORLD_CUP_FILE)

    try:
        df_results = pd.read_csv(results_path)
        df_elo = pd.read_csv(elo_path)
        df_world_cup = pd.read_csv(world_cup_path)
        print(f"Successfully loaded {RESULTS_FILE}, {ELO_RATINGS_FILE}, {WORLD_CUP_FILE}.")
        return df_results, df_elo, df_world_cup
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure CSVs are in the '{DATA_DIR}' directory.")
        raise

def verify_data_integrity(df_results: pd.DataFrame, df_elo: pd.DataFrame):
    """
    Verifies date ranges and row counts for the main dataframes.
    """
    # Convert date columns to datetime objects
    df_results[DATE_COL] = pd.to_datetime(df_results[DATE_COL], errors='coerce')
    df_elo[DATE_COL] = pd.to_datetime(df_elo[DATE_COL], errors='coerce') # Assuming ELO also has a 'date' column

    # Verify date range for results
    min_result_date = df_results[DATE_COL].min()
    max_result_date = df_results[DATE_COL].max()
    print(f"Results DataFrame date range: {min_result_date.strftime('%Y-%m-%d')} to {max_result_date.strftime('%Y-%m-%d')}.")
    if min_result_date < pd.to_datetime(MIN_DATE) or max_result_date > pd.to_datetime(MAX_DATE):
        print(f"Warning: Results date range ({min_result_date} - {max_result_date}) falls outside expected range ({MIN_DATE} - {MAX_DATE}).")

    # Verify row counts
    expected_match_count = 49000
    print(f"Results DataFrame row count: {len(df_results)}. Expected: ~{expected_match_count}.")
    if not (expected_match_count * 0.95 <= len(df_results) <= expected_match_count * 1.05):
        print(f"Warning: Results row count ({len(df_results)}) is significantly different from expected (~{expected_match_count}).")

    print(f"Elo Ratings DataFrame row count: {len(df_elo)}.")

    # You might want to add more specific checks for df_world_cup if needed
    return df_results, df_elo # Return processed DFs with datetime dates

if __name__ == '__main__':
    # Example usage
    try:
        results_df, elo_df, world_cup_df = load_dataframes()
        results_df, elo_df = verify_data_integrity(results_df, elo_df)
        # You can now pass these dataframes to the next cleaning step
    except Exception as e:
        print(f"Data ingestion failed: {e}")

