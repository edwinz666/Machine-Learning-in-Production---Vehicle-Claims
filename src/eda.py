# import db.db_model
# import sqlite3

# conn = sqlite3.connect('db/db.sqlite')
# for row in conn.execute('SELECT * FROM rent_apartments LIMIT 10'):
#     print(row)


# import pickle as pk

import pandas as pd
# from loguru import logger
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV, train_test_split

# from config import model_settings
# from model.pipeline.preparation import prepare_data

# import sqlite3

# import pandas as pd
# from loguru import logger
# from sqlalchemy import select

# from config import engine
# from db.db_model import RentApartments

# import re

# import pandas as pd
# from loguru import logger

# from model.pipeline.collection import load_data_from_db

# df = pd.read_csv("freMTPL2freq.csv")

# cols = ['Area', 'VehBrand', 'VehGas', 'Region'] # Age?
# # logger.info(f'encoding categorical columns: {cols}')

# df = pd.get_dummies(
#     df,
#     columns=cols,
#     drop_first=True,
# )
# print(df.columns)


feature_values = pd.DataFrame([{
    'VehPower': 5,
    'VehAge': 0,
    'DrivAge': 20,
    'BonusMalus': 100,
    'Exposure': 0.5,
    'Area': 'D',
    'VehBrand': 'B12',
    'VehGas': 'Regular',
    'Region': 'R82',
}])

print(feature_values)

