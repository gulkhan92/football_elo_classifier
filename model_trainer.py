# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss
import joblib # For saving/loading models
import os

from config import (
    LOGREG_PARAMS, ELASTICNET_PARAMS, LGBM_PARAMS,
    LOGREG_GRID, ELASTICNET_GRID, LGBM_GRID, MODELS_DIR
)

def train_and_tune_model(
    model_name: str,
    model_class,
    initial_params: dict,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list
) -> tuple:
    """
    Trains and tunes a model using GridSearchCV with TimeSeriesSplit.
    Evaluates on the validation set.
    """
    print(f"\n--- Training and Tuning {model_name} ---")

    # Combine train and validation for TimeSeriesSplit
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    # TimeSeriesSplit for chronological cross-validation
    # n_splits should be chosen carefully based on data size and time periods
    # For this project, we're using a fixed train/val split, so TimeSeriesSplit
    # would typically be applied *within* the training set if we were doing CV there.
    # For tuning on (train+val) and then evaluating on test, we can use a simpler approach
    # or apply TimeSeriesSplit to the combined train_val set for hyperparameter tuning.
    # Given the explicit train/val/test split, we'll tune on the validation set directly
    # after training on the training set, or use TimeSeriesSplit on the combined train+val
    # to find the best params, then retrain on all train+val.

    # Let's use TimeSeriesSplit on the combined train+val for robust hyperparameter tuning
    # This ensures that each fold maintains chronological order.
    # The splits will be (train_fold_1), (train_fold_1 + train_fold_2), etc.
    tscv = TimeSeriesSplit(n_splits=5) # Example: 5 splits

    # Initialize the model with initial parameters
    model = model_class(**initial_params)

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_log_loss', # Primary metric
        cv=tscv, # Use TimeSeriesSplit
        verbose=1,
        n_jobs=-1 # Use all available cores
    )

    # Fit GridSearchCV on the combined training and validation data
    # This means the best parameters are found by evaluating on chronological folds
    # within the train_val set.
    grid_search.fit(X_train_val, y_train_val)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_ # Convert neg_log_loss back to log_loss

    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best CV log-loss for {model_name}: {best_score:.4f}")

    # Retrain the best model on the entire training + validation set
    # This is the model that will be evaluated on the test set
    final_params = initial_params.copy()
    final_params.update(best_params)
    final_model = model_class(**final_params)
    final_model.fit(X_train_val, y_train_val)

    # Evaluate on the validation set (for comparison with other models)
    y_val_pred_proba = final_model.predict_proba(X_val)
    val_log_loss = log_loss(y_val, y_val_pred_proba)
    print(f"Validation log-loss for {model_name} (with best params): {val_log_loss:.4f}")

    return final_model, val_log_loss, best_params

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list,
    model_output_dir: str = MODELS_DIR
) -> dict:
    """
    Orchestrates the training and tuning of all specified models.
    Returns a dictionary of trained models and their validation log-losses.
    """
    trained_models = {}
    model_val_losses = {}

    # Ensure model output directory exists
    os.makedirs(model_output_dir, exist_ok=True)

    # Multinomial Logistic Regression
    logreg_model, logreg_val_loss, logreg_params = train_and_tune_model(
        "Multinomial Logistic Regression",
        LogisticRegression,
        LOGREG_PARAMS,
        LOGREG_GRID,
        X_train, y_train, X_val, y_val, feature_names
    )
    trained_models['LogisticRegression'] = logreg_model
    model_val_losses['LogisticRegression'] = logreg_val_loss
    joblib.dump(logreg_model, os.path.join(model_output_dir, 'logistic_regression_model.pkl'))

    # Elastic-Net Logistic Regression
    # Note: For Elastic-Net, LogisticRegression needs penalty='elasticnet' and solver='saga'
    elasticnet_model, elasticnet_val_loss, elasticnet_params = train_and_tune_model(
        "Elastic-Net Logistic Regression",
        LogisticRegression,
        ELASTICNET_PARAMS,
        ELASTICNET_GRID,
        X_train, y_train, X_val, y_val, feature_names
    )
    trained_models['ElasticNet'] = elasticnet_model
    model_val_losses['ElasticNet'] = elasticnet_val_loss
    joblib.dump(elasticnet_model, os.path.join(model_output_dir, 'elastic_net_model.pkl'))

    # LightGBM
    lgbm_model, lgbm_val_loss, lgbm_params = train_and_tune_model(
        "LightGBM",
        LGBMClassifier,
        LGBM_PARAMS,
        LGBM_GRID,
        X_train, y_train, X_val, y_val, feature_names
    )
    trained_models['LightGBM'] = lgbm_model
    model_val_losses['LightGBM'] = lgbm_val_loss
    joblib.dump(lgbm_model, os.path.join(model_output_dir, 'lightgbm_model.pkl'))

    # Select the best model based on validation log-loss
    best_model_name = min(model_val_losses, key=model_val_losses.get)
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected based on validation log-loss: {best_model_name} (Log-loss: {model_val_losses[best_model_name]:.4f})")
    joblib.dump(best_model, os.path.join(model_output_dir, 'best_model.pkl'))

    return trained_models, model_val_losses, best_model_name

if __name__ == '__main__':
    # Example usage (assuming X_train, y_train, X_val, y_val, feature_cols from Step 5)
    from data_ingestion import load_dataframes, verify_data_integrity
    from data_cleaning import clean_all_dataframes
    from feature_engineering import generate_features
    from data_splitter import chronological_split

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

        trained_models, val_losses, best_model_name = train_models(X_train, y_train, X_val, y_val, feature_cols)

        print("\nAll models trained and tuned.")
        print(f"Validation losses: {val_losses}")

    except Exception as e:
        print(f"Model training failed: {e}")

# model_trainer.py (continued)

# ... (previous imports and functions) ...

def train_binary_draw_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list,
    model_output_dir: str = 'models/'
) -> tuple:
    """
    Trains a binary classifier to predict if a match will be a draw.
    """
    print("\n--- Training Binary Draw Model ---")
    os.makedirs(model_output_dir, exist_ok=True)

    # Create binary target: 1 if draw, 0 otherwise
    y_train_binary = (y_train == 1).astype(int) # Assuming 1 is the 'Draw' class
    y_val_binary = (y_val == 1).astype(int)

    # Use LightGBM for the binary classifier as it's often strong
    # Or Logistic Regression as specified in the plan
    binary_model_class = LGBMClassifier
    binary_initial_params = {**LGBM_PARAMS, 'objective': 'binary', 'num_class': 1} # Adjust for binary
    binary_param_grid = LGBM_GRID # Can use the same grid or a simplified one

    # Combine train and validation for TimeSeriesSplit
    X_train_val_binary = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val_binary = pd.concat([y_train_binary, y_val_binary], ignore_index=True)

    tscv = TimeSeriesSplit(n_splits=5)

    binary_model = binary_model_class(**binary_initial_params)
    grid_search_binary = GridSearchCV(
        estimator=binary_model,
        param_grid=binary_param_grid,
        scoring='neg_log_loss', # For binary classification
        cv=tscv,
        verbose=0,
        n_jobs=-1
    )
    grid_search_binary.fit(X_train_val_binary, y_train_val_binary)

    best_binary_model = grid_search_binary.best_estimator_
    best_binary_params = grid_search_binary.best_params_
    best_binary_score = -grid_search_binary.best_score_

    print(f"Best parameters for Binary Draw Model: {best_binary_params}")
    print(f"Best CV log-loss for Binary Draw Model: {best_binary_score:.4f}")

    # Retrain on combined train+val
    final_binary_params = binary_initial_params.copy()
    final_binary_params.update(best_binary_params)
    final_binary_model = binary_model_class(**final_binary_params)
    final_binary_model.fit(X_train_val_binary, y_train_val_binary)

    y_val_pred_proba_binary = final_binary_model.predict_proba(X_val)[:, 1] # Probability of draw
    val_log_loss_binary = log_loss(y_val_binary, y_val_pred_proba_binary)
    print(f"Validation log-loss for Binary Draw Model: {val_log_loss_binary:.4f}")

    joblib.dump(final_binary_model, os.path.join(model_output_dir, 'binary_draw_model.pkl'))
    return final_binary_model, val_log_loss_binary

class TwoStagePredictor:
    """
    A class to perform two-stage predictions:
    1. Predict if it's a draw using a binary model.
    2. If not a draw, predict home/away win using a multiclass model.
    """
    def __init__(self, binary_draw_model, multiclass_model, draw_threshold: float = 0.5):
        self.binary_draw_model = binary_draw_model
        self.multiclass_model = multiclass_model
        self.draw_threshold = draw_threshold
        # Ensure multiclass model's classes are in order (0: Home Win, 1: Draw, 2: Away Win)
        # This is important for correctly mapping probabilities
        self.multiclass_classes = multiclass_model.classes_
        if not np.array_equal(self.multiclass_classes, [0, 1, 2]):
            raise ValueError("Multiclass model classes are not [0, 1, 2]. Please ensure consistent mapping.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates probability predictions using the two-stage approach.
        Returns probabilities for [Home Win, Draw, Away Win].
        """
        # Predict draw probability
        draw_proba = self.binary_draw_model.predict_proba(X)[:, 1] # Probability of class 1 (Draw)

        # Predict multiclass probabilities (Home Win, Draw, Away Win)
        multiclass_proba = self.multiclass_model.predict_proba(X)

        # Initialize final probabilities
        final_proba = np.zeros_like(multiclass_proba)

        for i in range(len(X)):
            if draw_proba[i] >= self.draw_threshold:
                # If binary model predicts draw, assign draw_proba to draw class
                # and distribute remaining (1 - draw_proba) to home/away based on multiclass model's ratio
                final_proba[i, 1] = draw_proba[i] # Probability of Draw
                remaining_proba = 1 - draw_proba[i]

                # Distribute remaining probability to Home Win (0) and Away Win (2)
                # based on their relative proportions from the multiclass model
                home_win_multiclass_ratio = multiclass_proba[i, 0] / (multiclass_proba[i, 0] + multiclass_proba[i, 2] + 1e-9) # Add epsilon to prevent division by zero
                away_win_multiclass_ratio = multiclass_proba[i, 2] / (multiclass_proba[i, 0] + multiclass_proba[i, 2] + 1e-9)

                final_proba[i, 0] = remaining_proba * home_win_multiclass_ratio
                final_proba[i, 2] = remaining_proba * away_win_multiclass_ratio
            else:
                # If binary model does not predict draw, assign 0 to draw class
                # and normalize home/away probabilities from multiclass model
                final_proba[i, 1] = 0 # No Draw
                home_away_sum = multiclass_proba[i, 0] + multiclass_proba[i, 2] + 1e-9
                final_proba[i, 0] = multiclass_proba[i, 0] / home_away_sum
                final_proba[i, 2] = multiclass_proba[i, 2] / home_away_sum

        # Ensure probabilities sum to 1 (due to potential floating point errors)
        final_proba = final_proba / final_proba.sum(axis=1, keepdims=True)
        return final_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates class predictions (0: Home Win, 1: Draw, 2: Away Win)
        using the two-stage approach.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

# model_evaluator.py (extended to include two-stage model evaluation)
# ... (previous imports and functions) ...

def evaluate_two_stage_model(
    binary_draw_model,
    multiclass_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    draw_threshold: float = 0.5,
    output_dir: str = 'reports/'
) -> dict:
    """
    Evaluates the two-stage prediction model on the test set.
    """
    print(f"\n--- Evaluating Two-Stage Model on Test Set (Threshold: {draw_threshold}) ---")
    os.makedirs(output_dir, exist_ok=True)

    two_stage_predictor = TwoStagePredictor(binary_draw_model, multiclass_model, draw_threshold)
    y_pred_proba = two_stage_predictor.predict_proba(X_test)
    y_pred = two_stage_predictor.predict(X_test)

    # 1. Multiclass Log-Loss
    test_log_loss = log_loss(y_test, y_pred_proba)
    print(f"Test Multiclass Log-Loss (Two-Stage): {test_log_loss:.4f}")

    # 2. Macro F1 Score
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Test Macro F1 Score (Two-Stage): {test_macro_f1:.4f}")

    # 3. Draw Recall (assuming 'Draw' is class 1)
    if 1 in y_test.unique() and 1 in multiclass_model.classes_:
        draw_recall = f1_score(y_test, y_pred, labels=[1], average='macro')
        print(f"Test Draw Recall (Two-Stage): {draw_recall:.4f}")
    else:
        draw_recall = 0.0
        print("Draw class (1) not present in test set or model classes, cannot calculate draw recall.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=multiclass_model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Confusion Matrix - Two-Stage Model (Threshold: {draw_threshold})')
    plt.savefig(os.path.join(output_dir, f'two_stage_model_confusion_matrix_t{str(draw_threshold).replace(".", "")}.png'))
    plt.close(fig)

    # Calibration Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, class_label in enumerate(multiclass_model.classes_):
        prob_true, prob_pred = calibration_curve(y_test == class_label, y_pred_proba[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=f'Class {class_label}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set_title(f'Calibration Curves - Two-Stage Model (Threshold: {draw_threshold})')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend()
    plt.savefig(os.path.join(output_dir, f'two_stage_model_calibration_curves_t{str(draw_threshold).replace(".", "")}.png'))
    plt.close(fig)

    metrics = {
        'model_name': f'Two-Stage (Threshold: {draw_threshold})',
        'test_log_loss': test_log_loss,
        'test_macro_f1': test_macro_f1,
        'test_draw_recall': draw_recall
    }
    return metrics

if __name__ == '__main__':
    # Example usage for two-stage model
    from data_ingestion import load_dataframes, verify_data_integrity
    from data_cleaning import clean_all_dataframes
    from feature_engineering import generate_features
    from data_splitter import chronological_split
    from model_trainer import train_models # To get best_model if not already saved

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

        # Train/Load multiclass model
        model_path_multiclass = 'models/best_model.pkl'
        if os.path.exists(model_path_multiclass):
            best_multiclass_model = joblib.load(model_path_multiclass)
            print(f"Loaded best multiclass model from {model_path_multiclass}")
        else:
            print("Best multiclass model not found, training models now...")
            trained_models, val_losses, best_model_name_from_train = train_models(X_train, y_train, X_val, y_val, feature_cols)
            best_multiclass_model = trained_models[best_model_name_from_train]

        # Train/Load binary draw model
        binary_draw_model_path = 'models/binary_draw_model.pkl'
        if os.path.exists(binary_draw_model_path):
            binary_draw_model = joblib.load(binary_draw_model_path)
            print(f"Loaded binary draw model from {binary_draw_model_path}")
        else:
            binary_draw_model, _ = train_binary_draw_model(X_train, y_train, X_val, y_val, feature_cols)

        # Evaluate the two-stage model
        two_stage_metrics = evaluate_two_stage_model(binary_draw_model, best_multiclass_model, X_test, y_test, draw_threshold=0.5)
        print("\nTwo-Stage Model Evaluation Metrics:")
        print(two_stage_metrics)

    except Exception as e:
        print(f"Two-stage model training/evaluation failed: {e}")
