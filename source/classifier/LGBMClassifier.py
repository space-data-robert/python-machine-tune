import optuna
import lightgbm as lgb


def lgbmclassifier():
    scoring = {
        'auc': 'roc_auc',
        'acc': 'accuracy',
        'recall': 'recall',
        'recall_macro': 'recall_macro',
        'recall_micro': 'recall_micro',
        'precision': 'precision',
        'precision_macro': 'precision_macro',
        'precision_micro': 'precision_micro',
        'f1': 'f1',
        'f1_macro': 'f1_macro',
        'f1_micro': 'f1_micro'
    }

    objective = {
        'binary': 'binary',
        'multiclass': 'multiclass',
    }

    model = lgb.LGBMClassifier(
        objective=objective['binary'],
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
        scoring=scoring['auc']
    )

    # search.trials_dataframe()
