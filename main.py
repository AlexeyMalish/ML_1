from typing import List

import category_encoders as ce
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from joblib import load


def transform_data(dataset):
    dataset.loc[dataset['max_power'] == ' bhp', 'max_power'] = np.nan
    dataset.loc[dataset['max_power'] == '0', 'max_power'] = np.nan
    dataset['mileage'] = dataset['mileage'].apply(lambda x: (
        float(x.split(' ')[0]) if x.split(' ')[1] == 'kmpl' else float(
            x.split(' ')[0]) * 1.4) if type(x) != float else x)
    dataset['engine'] = dataset['engine'].apply(lambda x: float(x.split(' ')[0]) if type(x) != float else x)
    dataset['max_power'] = dataset['max_power'].apply(lambda x: float(x.split(' ')[0]) if type(x) != float else x)
    dataset['name'] = dataset['name'].apply(lambda x: x.split(' ')[0])
    return dataset


def fill_NA(dataset, train_median_mileage, train_median_engine, train_median_max_power, train_median_seats):
    dataset['mileage'] = dataset['mileage'].fillna(train_median_mileage)
    dataset['engine'] = dataset['engine'].fillna(train_median_engine)
    dataset['max_power'] = dataset['max_power'].fillna(train_median_max_power)
    dataset['seats'] = dataset['seats'].fillna(train_median_seats)
    dataset['seats'] = dataset['seats'].astype('object')
    return dataset


df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
y_train = df_train['selling_price']
X_train = transform_data(df_train.drop(['selling_price', 'torque'], axis=1))
train_median_mileage = X_train['mileage'].median()
train_median_engine = X_train['engine'].median()
train_median_max_power = X_train['max_power'].median()
train_median_seats = X_train['seats'].median()

X_train['mileage'] = X_train['mileage'].fillna(train_median_mileage)
X_train['engine'] = X_train['engine'].fillna(train_median_engine)
X_train['max_power'] = X_train['max_power'].fillna(train_median_max_power)
X_train['seats'] = X_train['seats'].fillna(train_median_seats)
X_train['seats'] = X_train['seats'].astype('object')

tgt_enc = ce.TargetEncoder(cols=['name'], smoothing=15)
tgt_enc.fit(X_train, y_train)
X_train = tgt_enc.transform(X_train)

hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
hot_encoded_train = hot_encoder.fit_transform(X_train.select_dtypes('object'))
hot_encoded_train_df = pd.DataFrame(hot_encoded_train,
                                columns=hot_encoder.get_feature_names_out(
                                    list(X_train.select_dtypes('object').columns)))
X_train_hot_encoded = pd.concat([X_train.select_dtypes(exclude='object').reset_index(drop=True)
                                , hot_encoded_train_df.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
scaler.fit(X_train_hot_encoded)
X_train_scaler = pd.DataFrame(scaler.transform(X_train_hot_encoded), columns=X_train_hot_encoded.columns)

model = load('model.pkl')

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    input_data = pd.DataFrame([item.dict()]).drop(['selling_price', 'torque'], axis=1)

    input_data['mileage'] = input_data['mileage'].fillna(train_median_mileage)
    input_data['engine'] = input_data['engine'].fillna(train_median_engine)
    input_data['max_power'] = input_data['max_power'].fillna(train_median_max_power)
    input_data['seats'] = input_data['seats'].fillna(train_median_seats)
    input_data['seats'] = input_data['seats'].astype('object')

    data = transform_data(input_data)
    
    data = tgt_enc.transform(data)
    
    hot_encoded_data = hot_encoder.transform(data.select_dtypes('object'))
    hot_encoded_test_df = pd.DataFrame(hot_encoded_data,
                                   columns=hot_encoder.get_feature_names_out(data.select_dtypes('object').columns))
    data_cat_hot_encoded = pd.concat(
        [data.select_dtypes(exclude='object').reset_index(drop=True), hot_encoded_test_df.reset_index(drop=True)],
        axis=1)
    data_cat_hot_encoded_scal = pd.DataFrame(scaler.transform(data_cat_hot_encoded), columns=data_cat_hot_encoded.columns)
    
    predicted_price = model.predict(data_cat_hot_encoded_scal.values)
    return float(round(np.e ** predicted_price[0], 0))


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    input_data = items.dict()
    input_data = pd.DataFrame(input_data['objects'])
    input_data = input_data.drop(['selling_price', 'torque'], axis=1)

    input_data['mileage'] = input_data['mileage'].fillna(train_median_mileage)
    input_data['engine'] = input_data['engine'].fillna(train_median_engine)
    input_data['max_power'] = input_data['max_power'].fillna(train_median_max_power)
    input_data['seats'] = input_data['seats'].fillna(train_median_seats)
    input_data['seats'] = input_data['seats'].astype('object')

    data = transform_data(input_data)
    
    data = tgt_enc.transform(data)
    hot_encoded_data = hot_encoder.transform(data.select_dtypes('object'))
    hot_encoded_test_df = pd.DataFrame(hot_encoded_data,
                                   columns=hot_encoder.get_feature_names_out(data.select_dtypes('object').columns))
    data_cat_hot_encoded = pd.concat(
        [data.select_dtypes(exclude='object').reset_index(drop=True), hot_encoded_test_df.reset_index(drop=True)],
        axis=1)

    data_cat_hot_encoded_scal = pd.DataFrame(scaler.transform(data_cat_hot_encoded), columns=data_cat_hot_encoded.columns)

    predicted_prices = model.predict(data_cat_hot_encoded_scal.values)

    return [float(round(price, 0)) for price in np.e ** predicted_prices]

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
