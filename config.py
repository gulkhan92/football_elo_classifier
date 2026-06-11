# src/config.py

DATA_DIR = 'data/'
RESULTS_FILE = 'results.csv'
ELO_RATINGS_FILE = 'elo_ratings.csv'
WORLD_CUP_FILE = 'world_cup_matches.csv'

# Date ranges for data verification and splitting
MIN_DATE = '1872-01-01'
MAX_DATE = '2026-12-31' # Or a more precise end date if available

TRAIN_END_DATE = '2013-12-31'
VALIDATION_END_DATE = '2017-12-31'
TEST_START_DATE = '2018-01-01'

# Column names
DATE_COL = 'date'
HOME_TEAM_COL = 'home_team'
AWAY_TEAM_COL = 'away_team'
TEAM_COL = 'team' # For Elo ratings
SCORER_TEAM_COL = 'scorer_team' # If applicable in results
ELO_DATE_COL = 'rating_date' # For Elo ratings
ELO_RATING_COL = 'elo_rating'

# Model hyperparameters (initial suggestions, will be tuned)
LOGREG_PARAMS = {
    'penalty': 'l2',
    'solver': 'saga', # Supports elasticnet and l2
    'multi_class': 'multinomial',
    'max_iter': 1000,
    'random_state': 42
}

ELASTICNET_PARAMS = {
    'penalty': 'elasticnet',
    'solver': 'saga',
    'multi_class': 'multinomial',
    'max_iter': 1000,
    'random_state': 42
}

LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3, # Home Win, Draw, Away Win
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1
}

# Grid search parameters (example ranges)
LOGREG_GRID = {'C': [0.01, 0.1, 1, 10, 100]}
ELASTICNET_GRID = {'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
LGBM_GRID = {
    'num_leaves': [20, 31, 40],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'lambda_l1': [0, 0.1, 0.5],
    'lambda_l2': [0, 0.1, 0.5],
    'max_depth': [5, 7, 10]
}
