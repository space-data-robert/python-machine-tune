import optuna
import xgboost as xgb


def xgbregressor():
    return xgb.XGBRegressor(
        booster='gbtree',
        objective='reg:squarederror',
        random_state=27,
    )

def xgbregressor_params():
    prefix = 'xgbregressor'

    return {
        f'{prefix}__n_estimators': optuna.distributions.IntDistribution(1000, 10000),
        f'{prefix}__reg_alpha': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
        f'{prefix}__reg_lambda': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
        f'{prefix}__subsample': optuna.distributions.FloatDistribution(0.5, 1.0, step=0.1),
        f'{prefix}__learning_rate': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
        f'{prefix}__max_depth': optuna.distributions.IntDistribution(2, 9),
        f'{prefix}__colsample_bytree': optuna.distributions.FloatDistribution(0.1, 1.0),
    }
