import optuna
from sklearn.neighbors import KNeighborsClassifier


def kneighborsclassifier():
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

    model = KNeighborsClassifier(
        random_state=27,
        n_jobs=-1
    )

    params = {
        'n_estimators': optuna.distributions.IntDistribution(1, 60),

        'weights': optuna.distributions.CategoricalDistribution(choices=('uniform', 'distance')),
        'metric': optuna.distributions.CategoricalDistribution(choices=('euclidean', 'manhattan', 'minkowski')),
        'algorithm': optuna.distributions.CategoricalDistribution(choices=('auto', 'ball_tree', 'kd_tree', 'brute')),
        'leaf_size': optuna.distributions.IntDistribution(30, 60),
        'p': optuna.distributions.CategoricalDistribution(choices=(1, 2)),
    }

    search = optuna.integration.OptunaSearchCV(
        model,
        params,
        cv=3,
        n_trials=10**2,
        timeout=600,
        scoring=scoring['f1_macro']
    )

    # search.trials_dataframe()
