import dill
import os
import logging
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline

path = os.environ.get('PROJECT_PATH', '.')


def download() -> Pipeline:
    path_to_models = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{path_to_models[-1]}', 'rb') as file:
        model = dill.load(file)
    logging.info(f'Model is download')
    return model


def predict() -> None:
    prediction = pd.DataFrame(columns=['car_id', 'pred'])
    path_to_test = os.listdir(f'{path}/data/test')
    for i in range(len(path_to_test)):
        df = pd.read_json(f'{path}/data/test/{path_to_test[i]}', orient='index').T
        prediction.loc[len(prediction.index)] = [df['id'].iloc[0], ''.join(download().predict(df))]

    prediction.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)
    logging.info(f'Prediction is saved as preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
