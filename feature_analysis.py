# feature_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from config import REPORTS_DIR

def plot_feature_importance(model, feature_names: list, model_name: str, output_dir: str = REPORTS_DIR):
    """
    Extracts and plots feature importance for the given model.
    Handles both LightGBM (split importance) and Logistic Regression (coefficients).
    """
    print(f"\n--- Analyzing Feature Importance for {model_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    importance_df = pd.DataFrame({'feature': feature_names})

    if hasattr(model, 'feature_importances_'): # LightGBM
        importance_df['importance'] = model.feature_importances_
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        title = f'Feature Importance (Split) - {model_name}'
        print("Top 10 features by importance:")
        print(importance_df.head(10))

    elif hasattr(model, 'coef_'): # Logistic Regression
        # For multiclass, coef_ is (n_classes, n_features)
        # We can take the absolute mean across classes for overall importance
        importance_df['importance'] = np.mean(np.abs(model.coef_), axis=0)
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        title = f'Feature Importance (Absolute Mean Coefficients) - {model_name}'
        print("Top 10 features by absolute mean coefficient:")
        print(importance_df.head(10))

    else:
        print(f"Warning: Feature importance not available for model type {type(model)}.")
        return

    # Plotting feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20)) # Top 20 features
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png'))
    plt.close()
    print(f"Feature importance plot saved to {output_dir}")

    return importance_df

def plot_feature_family_importance(importance_df: pd.DataFrame, output_dir: str = REPORTS_DIR):
    """
    Groups features into families and plots their relative importance.
    Assumes feature names follow a pattern that allows grouping.
    """
    print("\n--- Analyzing Feature Family Importance ---")
    os.makedirs(output_dir, exist_ok=True)

    # Define patterns for feature families
    family_patterns = {
        'Elo Features': ['rating_diff', 'home_rating_pre_match', 'away_rating_pre_match', 'rating_age_days'],
        'Draw-Specific Features': ['abs_rating_diff', 'is_neutral_venue', 'is_world_cup', 'is_friendly', 'draw_rate_roll'],
        'Rolling Performance Features': ['points_roll', 'points_diff_roll'],
        'Attack-Defense Features': ['goals_scored_roll', 'goals_conceded_roll', 'goals_scored_diff_roll', 'goals_conceded_diff_roll']
    }

    family_importance = {family: 0.0 for family in family_patterns}

    for _, row in importance_df.iterrows():
        feature_name = row['feature']
        importance_value = row['importance']
        assigned = False
        for family, patterns in family_patterns.items():
            if any(pattern in feature_name for pattern in patterns):
                family_importance[family] += importance_value
                assigned = True
                break
        if not assigned:
            # Handle features that don't fit a defined family, or add them to an 'Other' category
            # print(f"Warning: Feature '{feature_name}' not assigned to any family.")
            pass

    family_importance_df = pd.DataFrame(list(family_importance.items()), columns=['Family', 'Total Importance'])
    family_importance_df = family_importance_df.sort_values('Total Importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Total Importance', y='Family', data=family_importance_df)
    plt.title('Relative Importance of Feature Families')
    plt.xlabel('Total Importance')
    plt.ylabel('Feature Family')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_family_importance.png'))
    plt.close()
    print(f"Feature family importance plot saved to {output_dir}")

    return family_importance_df

if __name__ == '__main__':
    # Example usage (assuming X_test, y_test from Step 5 and best_model from Step 6)
    from src.data_ingestion import load_dataframes, verify_data_integrity
    from src.data_cleaning import clean_all_dataframes
    from src.feature_engineering import generate_features
    from src.data_splitter import chronological_split
    from src.model_trainer import train_models # To get best_model if not already saved

    try:
        results_df, elo_df, world_cup_df = load_dataframes()
        results_df, elo_df = verify_data_integrity(results_df, elo_df)
        results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned = clean_all_dataframes(
            results_df, elo_df, world_cup_df
        )

        if DATE_COL in elo_df_cleaned.columns and ELO_DATE_COL not in elo_df_cleaned.columns:
            elo_df_cleaned = elo_df_cleaned.rename(columns={DATE_COL: ELO_DATE_COL})

        final_features_df = generate_features(results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned)

        result_mapping = {'Home Win': 0, 'Draw': 1, 'Away Win': 2}
        final_features_df['target'] = final_features_df['result'].map(result_mapping)

        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = chronological_split(final_features_df, 'target')

        # Load the best model
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            best_model = joblib.load(model_path)
            best_model_name = "Best Model"
            print(f"Loaded best model from {model_path}")
        else:
            print("Best model not found, training models now...")
            trained_models, val_losses, best_model_name_from_train = train_models(X_train, y_train, X_val, y_val, feature_cols)
            best_model = trained_models[best_model_name_from_train]
            best_model_name = best_model_name_from_train

        importance_df = plot_feature_importance(best_model, feature_cols, best_model_name)
        if importance_df is not None:
            plot_feature_family_importance(importance_df)

    except Exception as e:
        print(f"Feature analysis failed: {e}")
