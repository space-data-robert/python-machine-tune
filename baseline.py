import argparse
from sklearn.pipeline import make_pipeline

from source.common import *
from source.classifier import *
from source.regressor import *


def pipeline():
    pipeline = make_pipeline(

    )
    return pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_classifier', type=int, choices=range(2))
    args = parser.parse_args()

    x_data, y_data = read_csv(is_classifier=args.is_classifier)

    # pipeline()

