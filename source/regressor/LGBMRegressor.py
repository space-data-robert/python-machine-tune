import optuna
import lightgbm as lgb


def lgbmregressor():
    return lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        random_state=27,
    )

def lgbmregressor_params():
    prefix = 'lgbmregressor'
    params = {
        f'{prefix}__n_estimators': optuna.distributions.IntDistribution(1000, 10000),
        f'{prefix}__subsample': optuna.distributions.LogUniformDistribution(0.5, 1.0),
        f'{prefix}__learning_rate': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
        f'{prefix}__max_depth': optuna.distributions.IntDistribution(3, 15),
        f'{prefix}__min_child_samples': optuna.distributions.IntDistribution(5, 100),
        f'{prefix}__lambda_l1': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        f'{prefix}__lambda_l2': optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        f'{prefix}__num_leaves': optuna.distributions.IntDistribution(2, 256),
        f'{prefix}__feature_fraction': optuna.distributions.FloatDistribution(0.4, 1.0),
        f'{prefix}__bagging_fraction': optuna.distributions.FloatDistribution(0.4, 1.0),
        f'{prefix}__bagging_freq': optuna.distributions.IntDistribution(1, 7),
    }
    return params
