import autosklearn
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from sklearn.metrics import roc_auc_score
import PipelineProfiler


def automl(minutes: int=5) -> object:

    automl: object = AutoSklearn2Classifier(
        time_left_for_this_task=(60 * minutes),
        load_models=True,
        metric=autosklearn.metrics.roc_auc,
        memory_limit=(10 ** 4),
        seed=47,
    )
    return automl


# automl.fit(x_train, y_train)

# pred_proba = automl.predict_proba(x_test)[:, 1]

# auc_score = roc_auc_score(y_test, pred_proba)

# automl.leaderboard()

# automl_profile = PipelineProfiler.import_autosklearn(automl)

# PipelineProfiler.plot_pipeline_matrix(automl_profile)

