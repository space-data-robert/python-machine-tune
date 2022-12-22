import optuna
import catboost as cat


def catboostclassifier():
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
    params = {
        'learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.3),
        'bagging_temperature': optuna.distributions.LogUniformDistribution(0.01, 100.00),
        'n_estimators': optuna.distributions.IntDistribution(1000, 10000),
        'max_depth': optuna.distributions.IntDistribution(4, 16),
        'random_strength': optuna.distributions.IntDistribution(0, 100),
        'colsample_bylevel': optuna.distributions.FloatDistribution(0.4, 1.0),
        'l2_leaf_reg': optuna.distributions.FloatDistribution(1e-8, 3e-5),
        'min_child_samples': optuna.distributions.IntDistribution(5, 100),
        'max_bin': optuna.distributions.IntDistribution(200, 500),
        'od_type': optuna.distributions.CategoricalDistribution(['IncToDec', 'Iter']),
    }

    model = cat.CatBoostClassifier(
        random_state=27,
    )
    
    search = optuna.integration.OptunaSearchCV(
        model,
        params,
        cv=3,
        n_trials=10**2,
        timeout=600,
        scoring=scoring['auc']
    )

    # search.trials_dataframe()

