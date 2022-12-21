import argparse
from sklearn.pipeline import make_pipeline

from source.common import *
from source.classifier import *
from source.regressor import *


def pipeline(processor, x_data, y_data):
    pipeline = make_pipeline(
        processor,
        xgbregressor()
    )

    search = optuna_search(
        pipeline,
        xgbregressor_params(),
        'neg_mean_absolute_error'
    )
    search.fit(
        x_data,
        y_data
    )
    result = search.trials_dataframe()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regressor', type=int, choices=range(2))
    args = parser.parse_args()

    pipeline()

