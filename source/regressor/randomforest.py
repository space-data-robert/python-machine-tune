import optuna
from sklearn.ensemble import RandomForestRegressor


def randomforestregressor():
    scoring = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_percentage_error',
        'msle': 'neg_mean_squared_log_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
    }

    model = RandomForestRegressor(
        random_state=27,
    )

    params = {
        'n_estimators': optuna.distributions.IntDistribution(50, 1000),
        'max_depth': optuna.distributions.IntDistribution(4, 50),
        'min_samples_split': optuna.distributions.IntDistribution(1, 150),
        'min_samples_leaf': optuna.distributions.IntDistribution(1, 60),
    }

    search = optuna.integration.OptunaSearchCV(
        model,
        params,
        cv=3,
        n_trials=10**2,
        timeout=600,
        scoring=scoring['mse']
    )

    # search.trials_dataframe()
