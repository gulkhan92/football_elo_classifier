
# World Cup Prediction Machine Learning Pipeline

## Project Overview
This project develops a comprehensive machine learning pipeline to predict the outcomes of football (soccer) matches, with a particular focus on World Cup scenarios. It encompasses data collection, rigorous cleaning and feature engineering, model training with hyperparameter tuning, and a thorough evaluation of model performance, including specialized metrics for draw prediction.

The pipeline is designed to be modular and professional, emphasizing best practices such as preventing data leakage, using chronological data splits, and providing interpretable results.

## Features
The pipeline covers the following key steps:

1.  **Data Collection**: Loading and initial validation of match results, Elo ratings, and goalscorer data.
2.  **Data Cleaning & Standardization**: Standardizing team names across all datasets to ensure consistent merging and analysis.
3.  **Temporal Feature Engineering**:
    *   **Elo-based Features**: Incorporating pre-match Elo ratings, rating differences, and rating age, ensuring no future data leakage.
    *   **Draw-Specific Features**: Identifying neutral venues, World Cup matches, and friendly matches, along with rolling draw rates.
    *   **Rolling Performance Features**: Calculating points per match (win=3, draw=1, loss=0) over recent matches (e.g., last 5 and 10 games) for both home and away teams.
    *   **Attack & Defense Features**: Deriving goals scored and conceded per match over recent history for both teams.
4.  **Chronological Train-Validation-Test Split**: Dividing the dataset into distinct time-based periods to simulate real-world prediction scenarios and prevent data leakage.
5.  **Model Training & Tuning**: Training multiple machine learning models (Logistic Regression, Elastic-Net, LightGBM) using time-aware cross-validation and optimizing for multiclass log-loss.
6.  **Model Evaluation**: Comprehensive assessment of the best-performing model on an unseen test set, including log-loss, F1-score, draw recall, confusion matrices, and calibration curves.
7.  **Feature Importance & Interpretation**: Analyzing and visualizing the contribution of different feature families to the model's predictions.
8.  **Optional Two-Stage Draw Model**: Implementation and evaluation of a specialized binary classifier to improve draw prediction recall.

## Technologies Used
*   **Python**: The primary programming language.
*   **pandas**: For data manipulation and analysis.
*   **numpy**: For numerical operations.
*   **scikit-learn**: For machine learning models, cross-validation, and evaluation metrics.
*   **lightgbm**: For gradient boosting models.
*   **unidecode**: For handling character encoding and accents in team names.
*   **matplotlib & seaborn**: For data visualization and plotting.
*   **tqdm**: For progress bars during long-running operations.
*   **joblib**: For saving and loading trained models.

## Project Structure
```
football_elo_classifier/
├── data/
│   ├── results.csv             # Raw match results
│   ├── elo_ratings.csv         # Historical Elo ratings
│   └── goalscorers.csv         # Goalscorer data (used for World Cup/friendly flags)
├── src/
│   ├── config.py               # Centralized configuration (paths, dates, hyperparameters)
│   ├── data_ingestion.py       # Handles loading and initial data checks
│   ├── data_cleaning.py        # Functions for standardizing team names
│   ├── feature_engineering.py  # Logic for creating all features, including temporal joins
│   ├── data_splitter.py        # Manages chronological train/validation/test splits
│   ├── model_trainer.py        # Encapsulates model training, tuning, and the two-stage predictor
│   ├── model_evaluator.py      # Functions for evaluating models and generating plots
│   ├── feature_analysis.py     # For feature importance extraction and visualization
│   └── main.py                 # Orchestrates the entire ML pipeline
├── models/                     # Directory to store trained machine learning models
├── reports/                    # Directory for generated plots, metrics, and reports
├── .gitignore                  # Specifies files/directories to ignore in Git
├── requirements.txt            # Python dependencies
└── README.md                   # Project README file
```

## Setup
To get this project up and running on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/your-username/football_elo_classifier.git>
    cd football_elo_classifier
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # .\venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Acquire Data:**
    Download the necessary CSV files (`results.csv`, `elo_ratings.csv`, `goalscorers.csv`) and place them into the `data/` directory at the project root. These datasets are commonly available on platforms like Kaggle (e.g., "International Football Results from 1872 to 2026" and "Elo Ratings for International Football").

## Usage
To run the entire machine learning pipeline, execute the `main.py` script from the project root directory:

```bash
python -m src.main
```

The script will print progress and results to the console, and save generated models, plots, and detailed metrics into the `models/` and `reports/` directories, respectively.

## Results and Limitations
*(This section will be populated with actual results after running the pipeline. Refer to `reports/final_report.md` for detailed findings.)*

**Key Limitation**: Initial runs indicate a significant reduction in the dataset size after joining with Elo ratings, leading to very small validation and test sets. This can impact the reliability of hyperparameter tuning and model evaluation. Draw recall remains a challenging aspect, suggesting that current features might not fully capture the nuances leading to a draw.

## Future Work
*   **Enrich Elo Data**: Integrate a more comprehensive Elo rating dataset or implement a robust fallback mechanism for missing Elo ratings to retain more matches.
*   **Player-Level Data**: Incorporate player-specific data (injuries, individual form, suspensions) for potentially significant improvements, especially in predicting draws.
*   **Advanced Feature Engineering**: Explore more complex interaction features or external data sources (e.g., weather, home advantage factors beyond neutral venue).
*   **Model Interpretability**: Further deep dive into SHAP/LIME values for individual predictions.

## Acknowledgements
This project is inspired by common approaches in sports analytics and machine learning for predictive modeling.
```
<!--
[PROMPT_SUGGESTION]Can you help me find a more comprehensive Elo rating dataset or suggest a strategy to handle missing Elo ratings to retain more matches?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]How can I add unit tests for the feature_engineering.py module, specifically for the calculate_rolling_features function?[/PROMPT_SUGGESTION]
