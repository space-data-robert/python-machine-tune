import optuna
from sklearn.ensemble import IsolationForest


def isolationforest(the_contamination):
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
        'n_estimators': optuna.distributions.IntDistribution(500, 5000),
    }

    model = IsolationForest(
        random_state=27,
        contamination=the_contamination
    )
    
    search = optuna.integration.OptunaSearchCV(
        model,
        params,
        cv=3,
        n_trials=10**2,
        timeout=600,
        scoring=scoring['f1_macro']
    )
    # search.fit(x_train)
    
    # search.trials_dataframe()


