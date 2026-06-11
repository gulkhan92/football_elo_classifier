# src/data_cleaning.py
import pandas as pd
import re
from unidecode import unidecode
from config import HOME_TEAM_COL, AWAY_TEAM_COL, TEAM_COL, SCORER_TEAM_COL

def clean_team_name(name: str) -> str:
    """
    Cleans a team name by stripping whitespace, converting to lowercase,
    removing accents, and replacing non-alphanumeric characters with underscores.
    """
    if not isinstance(name, str):
        return name # Return as is if not a string (e.g., NaN)
    name = name.strip()
    name = name.lower()
    name = unidecode(name) # Remove accents
    name = re.sub(r'[^a-z0-9_]', '_', name) # Replace non-alphanumeric with underscore
    name = re.sub(r'_{2,}', '_', name) # Replace multiple underscores with a single one
    name = name.strip('_') # Remove leading/trailing underscores
    return name

def apply_team_name_cleaning(df: pd.DataFrame, columns_to_clean: list) -> pd.DataFrame:
    """
    Applies the clean_team_name function to specified columns in a DataFrame.
    """
    df_cleaned = df.copy()
    for col in columns_to_clean:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(clean_team_name)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df_cleaned

def clean_all_dataframes(df_results: pd.DataFrame, df_elo: pd.DataFrame, df_world_cup: pd.DataFrame):
    """
    Applies team name cleaning to all relevant columns across the dataframes.
    """
    # Define columns to clean for each DataFrame
    results_cols = [HOME_TEAM_COL, AWAY_TEAM_COL]
    elo_cols = [TEAM_COL]
    # Assuming world_cup_df also has home_team and away_team
    world_cup_cols = [HOME_TEAM_COL, AWAY_TEAM_COL]

    df_results_cleaned = apply_team_name_cleaning(df_results, results_cols)
    df_elo_cleaned = apply_team_name_cleaning(df_elo, elo_cols)
    df_world_cup_cleaned = apply_team_name_cleaning(df_world_cup, world_cup_cols)

    print("Team names cleaned across all specified dataframes.")
    return df_results_cleaned, df_elo_cleaned, df_world_cup_cleaned

def validate_team_names(df_results: pd.DataFrame, df_elo: pd.DataFrame, sample_size: int = 50):
    """
    Manually validates a random sample of matched team names between results and Elo.
    Prints samples for visual inspection.
    """
    # Get unique team names from both dataframes
    results_teams = pd.concat([df_results[HOME_TEAM_COL], df_results[AWAY_TEAM_COL]]).unique()
    elo_teams = df_elo[TEAM_COL].unique()

    # Find teams present in results but not in Elo (and vice-versa)
    teams_in_results_only = set(results_teams) - set(elo_teams)
    teams_in_elo_only = set(elo_teams) - set(results_teams)

    print(f"\n--- Team Name Validation (Sample Size: {sample_size}) ---")
    print(f"Total unique teams in Results: {len(results_teams)}")
    print(f"Total unique teams in Elo: {len(elo_teams)}")

    if teams_in_results_only:
        print(f"\nTeams in Results but not in Elo (sample of {min(sample_size, len(teams_in_results_only))}):")
        for team in list(teams_in_results_only)[:sample_size]:
            print(f"- {team}")
    else:
        print("\nAll teams in Results are also in Elo (based on unique names).")

    if teams_in_elo_only:
        print(f"\nTeams in Elo but not in Results (sample of {min(sample_size, len(teams_in_elo_only))}):")
        for team in list(teams_in_elo_only)[:sample_size]:
            print(f"- {team}")
    else:
        print("\nAll teams in Elo are also in Results (based on unique names).")

    # Further validation: check a random sample of matches
    print(f"\nRandom sample of {sample_size} matches from results_df for visual inspection:")
    sample_matches = df_results.sample(min(sample_size, len(df_results)), random_state=42)
    for _, row in sample_matches.iterrows():
        home_team = row[HOME_TEAM_COL]
        away_team = row[AWAY_TEAM_COL]
        print(f"Match: {home_team} vs {away_team}")
        # You might want to check if these teams exist in df_elo[TEAM_COL]
        # For example:
        # home_exists = home_team in elo_teams
        # away_exists = away_team in elo_teams
        # print(f"  Home in Elo: {home_exists}, Away in Elo: {away_exists}")


if __name__ == '__main__':
    # Example usage (assuming dataframes are loaded from Step 1)
    from src.data_ingestion import load_dataframes, verify_data_integrity
    try:
        results_df, elo_df, world_cup_df = load_dataframes()
        results_df, elo_df = verify_data_integrity(results_df, elo_df)

        results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned = clean_all_dataframes(
            results_df, elo_df, world_cup_df
        )
        validate_team_names(results_df_cleaned, elo_df_cleaned, sample_size=50)
    except Exception as e:
        print(f"Data cleaning failed: {e}")

