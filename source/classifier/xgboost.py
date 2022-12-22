import optuna
import xgboost as xgb


def xgbclassifier():
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
        'binary': 'binary:logistic',
        'multi_softmax': 'multi:softmax',
        'multi_softprob': 'multi:softprob'
    }

    model = xgb.XGBClassifier(
        objective=objective['binary'],
        booster='gbtree',
        random_state=27,
    )

    params = {
        'n_estimators': optuna.distributions.IntDistribution(1000, 10000),
        'reg_alpha': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
        'reg_lambda': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
        'subsample': optuna.distributions.FloatDistribution(0.5, 1.0, step=0.1),
        'learning_rate': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
        'max_depth': optuna.distributions.IntDistribution(2, 9),
        'colsample_bytree': optuna.distributions.FloatDistribution(0.1, 1.0)
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
