import optuna
import lightgbm as lgb


def lgbmregressor():
    scoring = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_percentage_error',
        'msle': 'neg_mean_squared_log_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
    }

    model = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        random_state=27,
    )

    params = {
        'n_estimators': optuna.distributions.IntDistribution(1000, 10000),
        'subsample': optuna.distributions.LogUniformDistribution(0.5, 1.0),
        'learning_rate': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
        'max_depth': optuna.distributions.IntDistribution(3, 15),
        'min_child_samples': optuna.distributions.IntDistribution(5, 100),
        'lambda_l1': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        'lambda_l2': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        'num_leaves': optuna.distributions.IntDistribution(2, 256),
        'feature_fraction': optuna.distributions.FloatDistribution(0.4, 1.0),
        'bagging_fraction': optuna.distributions.FloatDistribution(0.4, 1.0),
        'bagging_freq': optuna.distributions.IntDistribution(1, 7),
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
