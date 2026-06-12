# feature_engineering.py
import pandas as pd
import numpy as np
from config import DATE_COL, HOME_TEAM_COL, AWAY_TEAM_COL, TEAM_COL, ELO_DATE_COL, ELO_RATING_COL
from tqdm import tqdm # For progress bar

def perform_temporal_elo_join(df_results: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a temporal join to add pre-match Elo ratings to the results DataFrame.
    Ensures no future data leakage by only considering Elo ratings strictly before the match date.
    """
    df_results_copy = df_results.copy()
    df_elo_copy = df_elo.copy()

    # Ensure date columns are datetime and sorted
    df_results_copy[DATE_COL] = pd.to_datetime(df_results_copy[DATE_COL], errors='coerce')
    df_elo_copy[ELO_DATE_COL] = pd.to_datetime(df_elo_copy[ELO_DATE_COL], errors='coerce')

    # merge_asof requires non-null merge keys (on and by) on both sides
    df_results_copy = df_results_copy.dropna(subset=[DATE_COL, HOME_TEAM_COL, AWAY_TEAM_COL])
    df_elo_copy = df_elo_copy.dropna(subset=[ELO_DATE_COL, TEAM_COL])

    # Prepare Elo ratings for merging: rename columns for clarity
    # merge_asof requires the 'on' key (date) to be sorted globally.
    elo_home = df_elo_copy[[TEAM_COL, ELO_DATE_COL, ELO_RATING_COL]].rename(
        columns={TEAM_COL: HOME_TEAM_COL, ELO_DATE_COL: 'home_elo_date', ELO_RATING_COL: 'home_rating_pre_match'}
    ).sort_values('home_elo_date').reset_index(drop=True)

    elo_away = df_elo_copy[[TEAM_COL, ELO_DATE_COL, ELO_RATING_COL]].rename(
        columns={TEAM_COL: AWAY_TEAM_COL, ELO_DATE_COL: 'away_elo_date', ELO_RATING_COL: 'away_rating_pre_match'}
    ).sort_values('away_elo_date').reset_index(drop=True)

    # Use merge_asof for efficient temporal join
    # For home team
    df_results_copy = pd.merge_asof(
        df_results_copy.sort_values(DATE_COL),
        elo_home,
        left_on=DATE_COL,
        right_on='home_elo_date',
        by=HOME_TEAM_COL,
        direction='backward' # Finds the last row in 'right' whose 'on' key is less than or equal to the 'left' key
    )

    # For away team
    df_results_copy = pd.merge_asof(
        df_results_copy.sort_values(DATE_COL),
        elo_away,
        left_on=DATE_COL,
        right_on='away_elo_date',
        by=AWAY_TEAM_COL,
        direction='backward'
    )

    # Calculate rating age days
    df_results_copy['rating_age_days_home'] = (df_results_copy[DATE_COL] - df_results_copy['home_elo_date']).dt.days
    df_results_copy['rating_age_days_away'] = (df_results_copy[DATE_COL] - df_results_copy['away_elo_date']).dt.days

    # Calculate rating difference
    df_results_copy['rating_diff'] = df_results_copy['home_rating_pre_match'] - df_results_copy['away_rating_pre_match']

    # Drop rows where no prior rating exists (very early matches)
    initial_rows = len(df_results_copy)
    df_results_copy.dropna(subset=['home_rating_pre_match', 'away_rating_pre_match'], inplace=True)
    rows_dropped = initial_rows - len(df_results_copy)
    print(f"Dropped {rows_dropped} matches due to missing pre-match Elo ratings.")

    # Clean up temporary columns
    df_results_copy.drop(columns=['home_elo_date', 'away_elo_date'], inplace=True)

    print("Temporal Elo ratings successfully joined.")
    return df_results_copy.sort_values(DATE_COL).reset_index(drop=True) # Ensure final output is sorted by date

if __name__ == '__main__':
    # Example usage (assuming dataframes are loaded and cleaned)
    from data_ingestion import load_dataframes, verify_data_integrity
    from data_cleaning import clean_all_dataframes

    try:
        results_df, elo_df, world_cup_df = load_dataframes()
        results_df, elo_df = verify_data_integrity(results_df, elo_df)
        results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned = clean_all_dataframes(
            results_df, elo_df, world_cup_df
        )

        # Ensure Elo date column is correctly named for this step
        # Assuming df_elo has a 'date' column that needs to be renamed to ELO_DATE_COL
        if DATE_COL in elo_df_cleaned.columns and ELO_DATE_COL not in elo_df_cleaned.columns:
            elo_df_cleaned = elo_df_cleaned.rename(columns={DATE_COL: ELO_DATE_COL})

        df_with_elo = perform_temporal_elo_join(results_df_cleaned, elo_df_cleaned)
        print("\nDataFrame with Elo ratings head:")
        print(df_with_elo[[DATE_COL, HOME_TEAM_COL, AWAY_TEAM_COL, 'home_rating_pre_match', 'away_rating_pre_match', 'rating_diff', 'rating_age_days_home']].head())
        print(f"Total matches after Elo join: {len(df_with_elo)}")
    except Exception as e:
        print(f"Temporal Elo join failed: {e}")



def calculate_rolling_features(df: pd.DataFrame, window_sizes: list = [5, 10]) -> pd.DataFrame:
    """
    Calculates rolling performance, attack, and defense features for each team,
    ensuring no future data leakage.
    Features are calculated based on matches *before* the current match date.
    """
    df_processed = df.copy().sort_values(by=DATE_COL).reset_index(drop=True)

    # Define target mapping for points calculation
    # Assuming 'result' column exists with 'Home Win', 'Draw', 'Away Win'
    result_map = {'Home Win': 3, 'Draw': 1, 'Away Win': 0}
    df_processed['home_points'] = df_processed.apply(
        lambda row: result_map['Home Win'] if row['result'] == 'Home Win' else
                    result_map['Draw'] if row['result'] == 'Draw' else
                    result_map['Away Win'], axis=1
    )
    df_processed['away_points'] = df_processed.apply(
        lambda row: result_map['Home Win'] if row['result'] == 'Away Win' else # Away team wins when home team loses
                    result_map['Draw'] if row['result'] == 'Draw' else
                    result_map['Away Win'], axis=1
    )

    # Helper function to calculate rolling stats for a single team
    def get_team_rolling_stats(team_df: pd.DataFrame, window: int):
        # Sort by date to ensure correct rolling window
        team_df = team_df.sort_values(DATE_COL)

        # Calculate rolling sums/means for home matches
        team_df[f'home_goals_scored_roll_{window}'] = team_df['home_score'].shift(1).rolling(window=window, min_periods=1).mean()
        team_df[f'home_goals_conceded_roll_{window}'] = team_df['away_score'].shift(1).rolling(window=window, min_periods=1).mean()
        team_df[f'home_points_roll_{window}'] = team_df['home_points'].shift(1).rolling(window=window, min_periods=1).mean()
        team_df[f'home_draw_rate_roll_{window}'] = (team_df['result'] == 'Draw').shift(1).rolling(window=window, min_periods=1).mean()

        # Calculate rolling sums/means for away matches
        team_df[f'away_goals_scored_roll_{window}'] = team_df['away_score'].shift(1).rolling(window=window, min_periods=1).mean()
        team_df[f'away_goals_conceded_roll_{window}'] = team_df['home_score'].shift(1).rolling(window=window, min_periods=1).mean()
        team_df[f'away_points_roll_{window}'] = team_df['away_points'].shift(1).rolling(window=window, min_periods=1).mean()
        team_df[f'away_draw_rate_roll_{window}'] = (team_df['result'] == 'Draw').shift(1).rolling(window=window, min_periods=1).mean()

        return team_df

    # Create a combined DataFrame for easier rolling calculations per team
    # This involves duplicating rows for each team's perspective
    home_matches = df_processed[[DATE_COL, HOME_TEAM_COL, 'home_score', 'away_score', 'result', 'home_points', 'away_points']].copy()
    away_matches = df_processed[[DATE_COL, AWAY_TEAM_COL, 'away_score', 'home_score', 'result', 'away_points', 'home_points']].copy()

    home_matches.rename(columns={HOME_TEAM_COL: 'team', 'home_score': 'goals_scored', 'away_score': 'goals_conceded', 'home_points': 'points', 'away_points': 'opponent_points'}, inplace=True)
    away_matches.rename(columns={AWAY_TEAM_COL: 'team', 'away_score': 'goals_scored', 'home_score': 'goals_conceded', 'away_points': 'points', 'home_points': 'opponent_points'}, inplace=True)

    all_matches_per_team = pd.concat([home_matches, away_matches], ignore_index=True)
    all_matches_per_team = all_matches_per_team.sort_values(by=['team', DATE_COL]).reset_index(drop=True)

    # Apply rolling calculations using groupby and expanding/rolling
    # Using expanding for cumulative features, or rolling with shift(1) for past N matches
    for window in window_sizes:
        # For each team, calculate rolling stats based on past matches
        all_matches_per_team[f'goals_scored_roll_{window}'] = all_matches_per_team.groupby('team')['goals_scored'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        all_matches_per_team[f'goals_conceded_roll_{window}'] = all_matches_per_team.groupby('team')['goals_conceded'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        all_matches_per_team[f'points_roll_{window}'] = all_matches_per_team.groupby('team')['points'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        all_matches_per_team[f'draw_rate_roll_{window}'] = all_matches_per_team.groupby('team')['result'].transform(lambda x: (x.shift(1) == 'Draw').rolling(window=window, min_periods=1).mean())

    # Merge these rolling features back into the original df_processed
    # This requires careful merging to ensure home/away teams get their respective stats
    for window in window_sizes:
        # Home team features
        df_processed = pd.merge(
            df_processed,
            all_matches_per_team[[DATE_COL, 'team', f'goals_scored_roll_{window}', f'goals_conceded_roll_{window}', f'points_roll_{window}', f'draw_rate_roll_{window}']].rename(
                columns={'team': HOME_TEAM_COL,
                         f'goals_scored_roll_{window}': f'home_goals_scored_roll_{window}',
                         f'goals_conceded_roll_{window}': f'home_goals_conceded_roll_{window}',
                         f'points_roll_{window}': f'home_points_roll_{window}',
                         f'draw_rate_roll_{window}': f'home_draw_rate_roll_{window}'}),
            on=[DATE_COL, HOME_TEAM_COL],
            how='left'
        )
        # Away team features
        df_processed = pd.merge(
            df_processed,
            all_matches_per_team[[DATE_COL, 'team', f'goals_scored_roll_{window}', f'goals_conceded_roll_{window}', f'points_roll_{window}', f'draw_rate_roll_{window}']].rename(
                columns={'team': AWAY_TEAM_COL,
                         f'goals_scored_roll_{window}': f'away_goals_scored_roll_{window}',
                         f'goals_conceded_roll_{window}': f'away_goals_conceded_roll_{window}',
                         f'points_roll_{window}': f'away_points_roll_{window}',
                         f'draw_rate_roll_{window}': f'away_draw_rate_roll_{window}'}),
            on=[DATE_COL, AWAY_TEAM_COL],
            how='left'
        )

        # Calculate difference features
        df_processed[f'goals_scored_diff_roll_{window}'] = df_processed[f'home_goals_scored_roll_{window}'] - df_processed[f'away_goals_scored_roll_{window}']
        df_processed[f'goals_conceded_diff_roll_{window}'] = df_processed[f'home_goals_conceded_roll_{window}'] - df_processed[f'away_goals_conceded_roll_{window}']
        df_processed[f'points_diff_roll_{window}'] = df_processed[f'home_points_roll_{window}'] - df_processed[f'away_points_roll_{window}']
        df_processed[f'draw_rate_diff_roll_{window}'] = df_processed[f'home_draw_rate_roll_{window}'] - df_processed[f'away_draw_rate_roll_{window}']


    # Drop temporary columns used for rolling calculations
    df_processed.drop(columns=['home_points', 'away_points'], inplace=True)

    print("Rolling performance, attack, and defense features calculated.")
    return df_processed

def add_draw_specific_features(df: pd.DataFrame, df_world_cup: pd.DataFrame) -> pd.DataFrame:
    """
    Adds draw-specific features: neutral venue, World Cup flag, friendly match flag.
    """
    df_copy = df.copy()

    # Absolute rating difference
    df_copy['abs_rating_diff'] = abs(df_copy['rating_diff'])

    # Neutral venue (assuming a 'neutral' column exists in results.csv)
    # If not, you might need to infer it or get it from world_cup_df
    if 'neutral' in df_copy.columns:
        df_copy['is_neutral_venue'] = df_copy['neutral'].astype(int)
    else:
        # Placeholder if 'neutral' column is missing, you'd need to define logic
        print("Warning: 'neutral' column not found in results. Inferring from World Cup matches.")
        # This is a simplified inference. A more robust approach would be needed.
        world_cup_matches_set = set(df_world_cup[DATE_COL].astype(str) + df_world_cup[HOME_TEAM_COL] + df_world_cup[AWAY_TEAM_COL])
        df_copy['is_neutral_venue'] = df_copy.apply(
            lambda row: 1 if (row[DATE_COL].strftime('%Y-%m-%d') + row[HOME_TEAM_COL] + row[AWAY_TEAM_COL]) in world_cup_matches_set else 0,
            axis=1
        )


    # World Cup flag (assuming 'tournament' column exists and contains 'World Cup')
    if 'tournament' in df_copy.columns:
        df_copy['is_world_cup'] = df_copy['tournament'].apply(lambda x: 1 if 'world cup' in str(x).lower() else 0)
    else:
        print("Warning: 'tournament' column not found. Cannot create 'is_world_cup' feature directly.")
        # Fallback: use df_world_cup to mark World Cup matches
        world_cup_matches_set = set(df_world_cup[DATE_COL].astype(str) + df_world_cup[HOME_TEAM_COL] + df_world_cup[AWAY_TEAM_COL])
        df_copy['is_world_cup'] = df_copy.apply(
            lambda row: 1 if (row[DATE_COL].strftime('%Y-%m-%d') + row[HOME_TEAM_COL] + row[AWAY_TEAM_COL]) in world_cup_matches_set else 0,
            axis=1
        )


    # Friendly match flag (assuming 'tournament' column can identify friendlies)
    if 'tournament' in df_copy.columns:
        df_copy['is_friendly'] = df_copy['tournament'].apply(lambda x: 1 if 'friendly' in str(x).lower() else 0)
    else:
        print("Warning: 'tournament' column not found. Cannot create 'is_friendly' feature directly.")
        df_copy['is_friendly'] = 0 # Default to 0 if no info

    print("Draw-specific features added.")
    return df_copy

def generate_features(df_results: pd.DataFrame, df_elo: pd.DataFrame, df_world_cup: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the entire feature engineering process.
    """
    # Step 3: Temporal Join for Elo ratings
    df_with_elo = perform_temporal_elo_join(df_results, df_elo)

    # Step 4: Add draw-specific features
    df_with_draw_features = add_draw_specific_features(df_with_elo, df_world_cup)

    # Step 4: Calculate rolling features
    # Ensure 'result' column is present and correctly mapped for rolling features
    # Assuming 'result' column is already in df_results and contains 'Home Win', 'Draw', 'Away Win'
    # If not, you'd need to create it here based on home_score and away_score
    if 'home_score' in df_with_draw_features.columns and 'away_score' in df_with_draw_features.columns and 'result' not in df_with_draw_features.columns:
        def get_match_result(row):
            if row['home_score'] > row['away_score']:
                return 'Home Win'
            elif row['home_score'] < row['away_score']:
                return 'Away Win'
            else:
                return 'Draw'
        df_with_draw_features['result'] = df_with_draw_features.apply(get_match_result, axis=1)


    final_df = calculate_rolling_features(df_with_draw_features)

    # Drop any rows that might have NaNs from early rolling calculations if min_periods was not 1
    # Or if there are still NaNs from the initial Elo join
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    rows_dropped = initial_rows - len(final_df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows after feature engineering due to remaining NaNs.")

    print("All features engineered successfully.")
    return final_df

if __name__ == '__main__':
    # Example usage
    from data_ingestion import load_dataframes, verify_data_integrity
    from data_cleaning import clean_all_dataframes

    try:
        results_df, elo_df, world_cup_df = load_dataframes()
        results_df, elo_df = verify_data_integrity(results_df, elo_df)
        results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned = clean_all_dataframes(
            results_df, elo_df, world_cup_df
        )

        # Ensure Elo date column is correctly named for this step
        if DATE_COL in elo_df_cleaned.columns and ELO_DATE_COL not in elo_df_cleaned.columns:
            elo_df_cleaned = elo_df_cleaned.rename(columns={DATE_COL: ELO_DATE_COL})

        final_features_df = generate_features(results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned)
        print("\nFinal DataFrame with all features head:")
        print(final_features_df.head())
        print(f"Final DataFrame shape: {final_features_df.shape}")
        print(final_features_df.columns.tolist())
    except Exception as e:
        print(f"Feature engineering failed: {e}")
