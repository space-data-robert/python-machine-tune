import pandas as pd
import tensorflow as tf


def read_csv(is_classifier):
    target_name = 'survived'

    file_name = tf.keras.utils.get_file(
        'train.csv',
        origin='https://storage.googleapis.com/tf-datasets/titanic/train.csv'
    )
    x_data = pd.read_csv(file_name)

    if is_classifier is 0:
        target_name = 'fare'

    y_data = x_data.pop(
        target_name
    )
    return (x_data, y_data)

