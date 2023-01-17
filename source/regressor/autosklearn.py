import autosklearn.regression
import autosklearn.metrics
import PipelineProfiler


def automl(minutes: int=5) -> object:

    automl: object = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=(60 * minutes),
        load_models=True,
        metric=autosklearn.metrics.mean_absolute_error,
        memory_limit=(10 ** 4),
        seed=47,
    )
    return automl


# automl.fit(x_train, y_train)

# automl.leaderboard()

# automl_profile = PipelineProfiler.import_autosklearn(automl)

# PipelineProfiler.plot_pipeline_matrix(automl_profile)

