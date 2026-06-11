# src/data_splitter.py
import pandas as pd
from config import DATE_COL, TRAIN_END_DATE, VALIDATION_END_DATE, TEST_START_DATE

def chronological_split(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Splits the DataFrame into training, validation, and test sets chronologically.
    Ensures no future data leakage between splits.
    """
    df_sorted = df.sort_values(by=DATE_COL).reset_index(drop=True)

    train_df = df_sorted[df_sorted[DATE_COL] <= pd.to_datetime(TRAIN_END_DATE)]
    validation_df = df_sorted[
        (df_sorted[DATE_COL] > pd.to_datetime(TRAIN_END_DATE)) &
        (df_sorted[DATE_COL] <= pd.to_datetime(VALIDATION_END_DATE))
    ]
    test_df = df_sorted[df_sorted[DATE_COL] >= pd.to_datetime(TEST_START_DATE)]

    # Separate features (X) and target (y)
    feature_columns = [col for col in df_sorted.columns if col not in [target_column, DATE_COL, HOME_TEAM_COL, AWAY_TEAM_COL, 'home_score', 'away_score']]
    # Add other columns to exclude from features if they are not features themselves
    # e.g., 'tournament', 'city', 'country', 'neutral' if they are not encoded as features

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_val = validation_df[feature_columns]
    y_val = validation_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    print(f"Data split chronologically:")
    print(f"  Train set: {len(train_df)} matches (up to {TRAIN_END_DATE})")
    print(f"  Validation set: {len(validation_df)} matches ({pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1)} to {VALIDATION_END_DATE})")
    print(f"  Test set: {len(test_df)} matches (from {TEST_START_DATE} onwards)")

    # Verify no date leakage
    if not train_df[DATE_COL].max() < validation_df[DATE_COL].min():
        print("Warning: Date leakage detected between train and validation sets!")
    if not validation_df[DATE_COL].max() < test_df[DATE_COL].min():
        print("Warning: Date leakage detected between validation and test sets!")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns

if __name__ == '__main__':
    # Example usage (assuming final_features_df from Step 4)
    from src.data_ingestion import load_dataframes, verify_data_integrity
    from src.data_cleaning import clean_all_dataframes
    from src.feature_engineering import generate_features

    try:
        results_df, elo_df, world_cup_df = load_dataframes()
        results_df, elo_df = verify_data_integrity(results_df, elo_df)
        results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned = clean_all_dataframes(
            results_df, elo_df, world_cup_df
        )

        if DATE_COL in elo_df_cleaned.columns and ELO_DATE_COL not in elo_df_cleaned.columns:
            elo_df_cleaned = elo_df_cleaned.rename(columns={DATE_COL: ELO_DATE_COL})

        final_features_df = generate_features(results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned)

        # Define the target column (e.g., 'result' from feature_engineering)
        # Ensure 'result' column is numerical for modeling (e.g., 0 for Home Win, 1 for Draw, 2 for Away Win)
        # This mapping should be consistent throughout the project.
        result_mapping = {'Home Win': 0, 'Draw': 1, 'Away Win': 2}
        final_features_df['target'] = final_features_df['result'].map(result_mapping)

        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = chronological_split(final_features_df, 'target')

        print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Shape of X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"Number of features: {len(feature_cols)}")

    except Exception as e:
        print(f"Data splitting failed: {e}")

