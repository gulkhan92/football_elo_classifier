# main.py
import pandas as pd
import os
import sys
import joblib

# This ensures the script can be run directly (e.g., 'python main.py')
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import DATE_COL, ELO_DATE_COL, DATA_DIR, MODELS_DIR, REPORTS_DIR
from data_ingestion import load_dataframes, verify_data_integrity
from data_cleaning import clean_all_dataframes
from feature_engineering import generate_features
from data_splitter import chronological_split
from model_trainer import train_models, train_binary_draw_model
from model_evaluator import evaluate_model, evaluate_two_stage_model
from feature_analysis import plot_feature_importance, plot_feature_family_importance

def run_pipeline():
    """
    Orchestrates the entire machine learning pipeline.
    """
    print("Starting World Cup Prediction ML Pipeline...\n")

    # --- Step 1: Data Collection ---
    print("Step 1: Data Collection")
    results_df, elo_df, world_cup_df = load_dataframes()
    results_df, elo_df = verify_data_integrity(results_df, elo_df)
    print("-" * 50)

    # --- Step 2: Data Cleaning and Standardization ---
    print("Step 2: Data Cleaning and Standardization")
    results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned = clean_all_dataframes(
        results_df, elo_df, world_cup_df
    )
    # Ensure Elo date column is correctly named for feature engineering
    if DATE_COL in elo_df_cleaned.columns and ELO_DATE_COL not in elo_df_cleaned.columns:
        elo_df_cleaned = elo_df_cleaned.rename(columns={DATE_COL: ELO_DATE_COL})
    print("-" * 50)

    # --- Step 3 & 4: Feature Engineering ---
    print("Step 3 & 4: Feature Engineering (Temporal Join & Feature Creation)")
    final_features_df = generate_features(results_df_cleaned, elo_df_cleaned, world_cup_df_cleaned)
    print(f"Final DataFrame shape after feature engineering: {final_features_df.shape}")
    print("-" * 50)

    # Prepare target variable
    result_mapping = {'Home Win': 0, 'Draw': 1, 'Away Win': 2}
    final_features_df['target'] = final_features_df['result'].map(result_mapping)

    # --- Step 5: Train-Validation-Test Split ---
    print("Step 5: Train-Validation-Test Split (Chronological)")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = chronological_split(final_features_df, 'target')
    print("-" * 50)

    # --- Step 6: Model Training ---
    print("Step 6: Model Training and Hyperparameter Tuning")
    trained_models, val_losses, best_model_name = train_models(X_train, y_train, X_val, y_val, feature_cols)
    best_multiclass_model = trained_models[best_model_name]
    print("-" * 50)

    # --- Step 7: Evaluation on Test Set (Best Multiclass Model) ---
    print("Step 7: Evaluation on Test Set (Best Multiclass Model)")
    best_model_metrics = evaluate_model(best_multiclass_model, X_test, y_test, best_model_name)
    print("-" * 50)

    # --- Step 8: Feature Importance and Interpretation ---
    print("Step 8: Feature Importance and Interpretation")
    importance_df = plot_feature_importance(best_multiclass_model, feature_cols, best_model_name)
    if importance_df is not None:
        plot_feature_family_importance(importance_df)
    print("-" * 50)

    # --- Step 9: Optional Draw-Specific Model ---
    print("Step 9: Optional Draw-Specific Model")
    binary_draw_model, binary_val_loss = train_binary_draw_model(X_train, y_train, X_val, y_val, feature_cols)
    two_stage_metrics = evaluate_two_stage_model(binary_draw_model, best_multiclass_model, X_test, y_test, draw_threshold=0.5)
    print("-" * 50)

    # --- Step 10: Final Reporting ---
    print("Step 10: Final Reporting")
    print("\n--- Summary of Results ---")
    print(f"Best Multiclass Model: {best_model_name}")
    print(f"Validation Log-Loss ({best_model_name}): {val_losses[best_model_name]:.4f}")
    print(f"Test Log-Loss ({best_model_name}): {best_model_metrics['test_log_loss']:.4f}")
    print(f"Test Macro F1 ({best_model_name}): {best_model_metrics['test_macro_f1']:.4f}")
    print(f"Test Draw Recall ({best_model_name}): {best_model_metrics['test_draw_recall']:.4f}")

    print("\nTwo-Stage Model Performance:")
    print(f"Test Log-Loss (Two-Stage): {two_stage_metrics['test_log_loss']:.4f}")
    print(f"Test Macro F1 (Two-Stage): {two_stage_metrics['test_macro_f1']:.4f}")
    print(f"Test Draw Recall (Two-Stage): {two_stage_metrics['test_draw_recall']:.4f}")

    print("\n--- Conclusions ---")
    print("The final model selection is expected to be LightGBM, given its typical performance on tabular data.")
    print(f"The best performing model on the validation set was: {best_model_name}.")
    print(f"Its test log-loss was {best_model_metrics['test_log_loss']:.4f}.")
    print(f"A key limitation observed is that draw recall remains relatively low ({best_model_metrics['test_draw_recall']:.2f} for the best multiclass model and {two_stage_metrics['test_draw_recall']:.2f} for the two-stage model).")
    print("This suggests that current features might not fully capture the nuances leading to a draw.")
    print("For meaningful improvement, incorporating player-level data (injuries, individual scoring form, suspensions) is likely necessary, rather than solely relying on more complex algorithms or further feature engineering from aggregate team statistics.")
    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    run_pipeline()
