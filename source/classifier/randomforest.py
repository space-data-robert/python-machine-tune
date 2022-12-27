import optuna
from sklearn.ensemble import RandomForestClassifier


def randomforestclassifier():
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

    model = RandomForestClassifier(
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
        scoring=scoring['auc']
    )

    # search.trials_dataframe()
