import optuna
import xgboost as xgb


def xgbregressor():
    scoring = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_percentage_error',
        'msle': 'neg_mean_squared_log_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
    }
    params = {
        'n_estimators': optuna.distributions.IntDistribution(1000, 10000),
        'reg_alpha': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
        'reg_lambda': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
        'subsample': optuna.distributions.FloatDistribution(0.5, 1.0, step=0.1),
        'learning_rate': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
        'max_depth': optuna.distributions.IntDistribution(2, 9),
        'colsample_bytree': optuna.distributions.FloatDistribution(0.1, 1.0)
    }

    model = xgb.XGBRegressor(
        objective='reg:linear',
        booster='gbtree',
        random_state=27,
    )
    search = optuna.integration.OptunaSearchCV(
        model,
        params,
        cv=3,
        n_trials=10**2,
        timeout=600,
        scoring=scoring['mse']
    )

    # search.trials_dataframe()
