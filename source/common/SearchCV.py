import optuna


def optuna_search(pipeline, params, scoring):
    return optuna.integration.OptunaSearchCV(
        pipeline,
        params,
        cv=3,
        n_trials=10**2,
        timeout=600,
        scoring=scoring,
    )