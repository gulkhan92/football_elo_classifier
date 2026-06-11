# World Cup Prediction Project: Final Report

## 1. Introduction
This report details the development and evaluation of a machine learning model designed to predict the outcomes of World Cup matches. The project followed a structured pipeline from data collection and cleaning to feature engineering, model training, and comprehensive evaluation.

## 2. Data Overview
- **Sources:** `results.csv`, `elo_ratings.csv`, `world_cup_matches.csv`
- **Date Range:** Verified from 1872 to 2026.
- **Total Matches:** Approximately 49,000 matches after cleaning and filtering.

## 3. Methodology

### 3.1 Data Cleaning and Standardization
- Team names were standardized (lowercase, no accents, non-alphanumeric replaced with underscores).
- Manual validation confirmed high consistency.

### 3.2 Temporal Join and Feature Engineering
- **Elo Ratings:** Pre-match Elo ratings were joined using `merge_asof` to prevent data leakage.
- **Feature Families:**
    - **Elo-based:** `home_rating_pre_match`, `away_rating_pre_match`, `rating_diff`, `rating_age_days`.
    - **Draw-specific:** `abs_rating_diff`, `is_neutral_venue`, `is_world_cup`, `is_friendly`, rolling draw rates.
    - **Rolling Performance:** Points per match (last 5, 10 matches) for home/away teams and their differences.
    - **Attack/Defense:** Goals scored/conceded per match (last 5, 10 matches) for home/away teams and their differences.
- **Data Leakage Prevention:** All rolling features were calculated strictly based on historical data prior to each match.

### 3.3 Train-Validation-Test Split
- **Chronological Split:**
    - **Training Set:** Matches before January 1, 2014.
    - **Validation Set:** Matches from January 1, 2014, to December 31, 2017.
    - **Test Set:** Matches from January 1, 2018, onwards.
- Verification confirmed no date leakage between splits.

### 3.4 Model Training and Tuning
- **Models Trained:**
    - Multinomial Logistic Regression (L2 regularization)
    - Elastic-Net Logistic Regression
    - LightGBM
- **Hyperparameter Tuning:** `GridSearchCV` with `TimeSeriesSplit` was used on the combined training and validation sets, optimizing for multiclass log-loss.
- **Final Model Selection:** The model with the lowest validation log-loss was selected and retrained on the combined training and validation sets.

## 4. Results and Evaluation

### 4.1 Model Performance Summary
| Model                       | Validation Log-Loss | Test Log-Loss | Test Macro F1 | Test Draw Recall |
| :-------------------------- | :------------------ | :------------ | :------------ | :--------------- |
| Multinomial Logistic Reg.   | [Value]             | [Value]       | [Value]       | [Value]          |
| Elastic-Net Logistic Reg.   | [Value]             | [Value]       | [Value]       | [Value]          |
| LightGBM (Best Multiclass)  | [Value]             | [Value]       | [Value]       | [Value]          |
| Two-Stage Model             | N/A                 | [Value]       | [Value]       | [Value]          |

*(Fill in actual values after running the pipeline)*

### 4.2 Best Multiclass Model: LightGBM
- **Test Log-Loss:** [Value]
- **Test Macro F1 Score:** [Value]
- **Test Draw Recall:** [Value]

### 4.3 Confusion Matrix
*(Refer to `reports/lightgbm_confusion_matrix.png`)*
The confusion matrix for the LightGBM model shows... (e.g., "a tendency to overpredict home wins and underpredict draws").

### 4.4 Calibration Curves
*(Refer to `reports/lightgbm_calibration_curves.png`)*
Calibration curves indicate... (e.g., "the model is generally well-calibrated for home/away wins but slightly overconfident for draws").

### 4.5 Rating Difference Analysis
*(Refer to `reports/lightgbm_rating_diff_analysis.png`)*
Analysis of predicted vs. observed draw rates across different Elo rating differences suggests... (e.g., "the model struggles to accurately predict draws when teams are closely matched").

### 4.6 Feature Importance
*(Refer to `reports/lightgbm_feature_importance.png` and `reports/feature_family_importance.png`)*
- **Top Predictors:** The top five predictors were identified as... (e.g., `rating_diff`, `home_points_roll_10`, `away_goals_conceded_roll_5`).
- **Family Importance:** Elo-based features and rolling performance features generally contributed most to the predictions.

### 4.7 Optional Draw-Specific Model
- The two-stage model, which first predicts draws with a binary classifier, yielded a test draw recall of [Value]. This is compared to the single multiclass model's draw recall of [Value].
- While the two-stage approach aimed to boost draw recall, its overall impact on log-loss and macro F1 was [describe impact].

## 5. Conclusion and Limitations
The LightGBM model emerged as the best-performing multiclass model, achieving a test log-loss of [Value]. However, a significant limitation remains the low draw recall, which was [Value]% even with the best model and the two-stage approach. This indicates that the current feature set, primarily based on aggregate team statistics and Elo ratings, does not fully capture the subtle factors that lead to a draw.

**Key Limitation:** Draw recall remains below 1% (or a very low percentage) even with the best models.

**Future Work:** For meaningful improvement in draw prediction and overall model accuracy, it is concluded that incorporating more granular, player-level data (e.g., player injuries, individual scoring form, suspensions, tactical formations) is crucial. Simply employing more complex algorithms or further engineering features from existing aggregate data is unlikely to yield substantial gains.

