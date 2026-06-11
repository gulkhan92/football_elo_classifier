# src/model_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, output_dir: str = 'reports/') -> dict:
    """
    Evaluates a trained model on the test set and generates various metrics and plots.
    """
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    os.makedirs(output_dir, exist_ok=True)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # 1. Multiclass Log-Loss
    test_log_loss = log_loss(y_test, y_pred_proba)
    print(f"Test Multiclass Log-Loss: {test_log_loss:.4f}")

    # 2. Macro F1 Score
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Test Macro F1 Score: {test_macro_f1:.4f}")

    # 3. Draw Recall (assuming 'Draw' is class 1)
    # Check if 'Draw' class exists in y_test and y_pred
    if 1 in y_test.unique() and 1 in model.classes_:
        draw_recall = f1_score(y_test, y_pred, labels=[1], average='macro') # F1 for draw class
        print(f"Test Draw Recall: {draw_recall:.4f}")
    else:
        draw_recall = 0.0
        print("Draw class (1) not present in test set or model classes, cannot calculate draw recall.")


    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close(fig)
    print(f"Confusion matrix saved to {output_dir}")

    # 5. Class-specific Calibration Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, class_label in enumerate(model.classes_):
        prob_true, prob_pred = calibration_curve(y_test == class_label, y_pred_proba[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=f'Class {class_label}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set_title(f'Calibration Curves - {model_name}')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_calibration_curves.png'))
    plt.close(fig)
    print(f"Calibration curves saved to {output_dir}")

    # 6. Rating-Difference Analysis (requires 'rating_diff' in X_test)
    if 'rating_diff' in X_test.columns:
        df_test_with_preds = X_test.copy()
        df_test_with_preds['true_target'] = y_test
        df_test_with_preds['pred_proba_draw'] = y_pred_proba[:, 1] # Assuming draw is class 1

        # Bin matches by absolute Elo difference
        bins = np.arange(0, X_test['rating_diff'].abs().max() + 50, 50)
        df_test_with_preds['rating_diff_bin'] = pd.cut(df_test_with_preds['rating_diff'].abs(), bins, right=False)

        # Compare predicted draw probability to observed draw rate in each bin
        rating_diff_analysis = df_test_with_preds.groupby('rating_diff_bin').agg(
            observed_draw_rate=('true_target', lambda x: (x == 1).mean()),
            mean_predicted_draw_proba=('pred_proba_draw', 'mean'),
            num_matches=('true_target', 'count')
        ).reset_index()

        print("\nRating Difference Analysis (Predicted vs. Observed Draw Rate):")
        print(rating_diff_analysis)

        fig, ax = plt.subplots(figsize=(12, 7))
        rating_diff_analysis.plot(x='rating_diff_bin', y=['observed_draw_rate', 'mean_predicted_draw_proba'], kind='bar', ax=ax)
        ax.set_title(f'Rating Difference Analysis - {model_name}')
        ax.set_xlabel('Absolute Elo Difference Bin')
        ax.set_ylabel('Rate / Probability')
        ax.legend(['Observed Draw Rate', 'Mean Predicted Draw Probability'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_rating_diff_analysis.png'))
        plt.close(fig)
        print(f"Rating difference analysis plot saved to {output_dir}")
    else:
        print("Skipping Rating Difference Analysis: 'rating_diff' column not found in X_test.")

    metrics = {
        'model_name': model_name,
        'test_log_loss': test_log_loss,
        'test_macro_f1': test_macro_f1,
        'test_draw_recall': draw_recall
    }
    return metrics

if __name__ == '__main__':
    # Example usage (assuming X_test, y_test from Step 5 and best_model from Step 6)
    from src.data_ingestion import load_dataframes, verify_data_integrity
    from src.data_cleaning import clean_all_dataframes
    from src.feature_engineering import generate_features
    from src.data_splitter import chronological_split
    from src.model_trainer import train_models
    import joblib

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

        # Load the best model (or train if not already done)
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            best_model = joblib.load(model_path)
            best_model_name = "Best Model (from file)"
            print(f"Loaded best model from {model_path}")
        else:
            print("Best model not found, training models now...")
            trained_models, val_losses, best_model_name_from_train = train_models(X_train, y_train, X_val, y_val, feature_cols)
            best_model = trained_models[best_model_name_from_train]
            best_model_name = best_model_name_from_train


        evaluation_metrics = evaluate_model(best_model, X_test, y_test, best_model_name)
        print("\nEvaluation Metrics:")
        print(evaluation_metrics)

    except Exception as e:
        print(f"Model evaluation failed: {e}")

