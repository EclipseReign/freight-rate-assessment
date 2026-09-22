from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

CAT = ['pickup', 'delivery', 'equipment']


def features(frame, market=False, signal=False):
    d = frame.copy()
    x = d[CAT].fillna('Unknown').astype(str).copy()
    distance = pd.to_numeric(d.distance, errors='raise')
    if (distance <= 0).any():
        raise ValueError('distance must be positive')
    x['log_distance'] = np.log(distance)
    x['inverse_distance'] = 1 / distance
    x['distance'] = distance
    weight = pd.to_numeric(d.weight, errors='coerce')
    x['weight'] = weight.where(weight > 0)
    x['weight_missing'] = x.weight.isna().astype(float)
    date = pd.to_datetime(d.date, errors='raise')
    day = (date - pd.Timestamp('2025-01-01')).dt.days
    x['trend'] = day / 365.25
    for period, label in [(365.25, 'annual'), (7, 'weekly')]:
        for k in [1, 2]:
            x[f'{label}_sin{k}'] = np.sin(2 * np.pi * k * day / period)
            x[f'{label}_cos{k}'] = np.cos(2 * np.pi * k * day / period)
    for col in ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
        x[col] = pd.to_numeric(d[col], errors='coerce')
    x['latitude_delta'] = x.delivery_lat - x.pickup_lat
    x['longitude_delta'] = x.delivery_lon - x.pickup_lon
    if market:
        x['market_index'] = pd.to_numeric(d.market_index, errors='coerce')
    if signal:
        x['quote_signal'] = pd.to_numeric(d.quote_signal, errors='coerce')
    return x


class FreightModel:
    def __init__(self, kind='ridge', market=False, signal=False, simple_time=False):
        self.kind, self.market, self.signal = kind, market, signal
        self.simple_time = simple_time

    def prepare(self, frame, fit=False):
        x = features(frame, self.market, self.signal)
        if self.simple_time:
            x = x.drop(columns=['trend', 'annual_sin2', 'annual_cos2'])
        numeric = list(x.select_dtypes('number').columns)
        if fit:
            self.medians = x[numeric].median().fillna(0)
        x[numeric] = x[numeric].fillna(self.medians)
        return x

    def fit(self, frame):
        x = self.prepare(frame, fit=True)
        y = frame.posted_rate.to_numpy() / frame.distance.to_numpy()
        if self.kind == 'ridge':
            numeric = list(x.select_dtypes('number').columns)
            self.encoder = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT),
                ('num', StandardScaler(), numeric),
            ])
            a = self.encoder.fit_transform(x)
            # Iteratively reweighted ridge: retain every row while limiting influence
            # of extreme responses. No validation response enters this calculation.
            self.model = Ridge(alpha=20.0)
            weights = np.ones(len(y))
            for _ in range(15):
                self.model.fit(a, y, sample_weight=weights)
                residual = y - self.model.predict(a)
                scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), .01)
                weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-9))
        else:
            self.model = CatBoostRegressor(iterations=700, depth=6, learning_rate=.055,
                loss_function='MAE', random_seed=42, thread_count=6,
                verbose=False, allow_writing_files=False, l2_leaf_reg=5)
            self.model.fit(x, y, cat_features=CAT)
        return self

    def predict(self, frame):
        x = self.prepare(frame)
        a = self.encoder.transform(x) if self.kind == 'ridge' else x
        return np.maximum(self.model.predict(a) * frame.distance.to_numpy(), .01)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    error = pred - y
    return dict(n=len(y), mae=float(np.mean(abs(error))),
        rmse=float(np.sqrt(np.mean(error ** 2))),
        wape=float(np.sum(abs(error)) / np.sum(abs(y))),
        median_ae=float(np.median(abs(error))), bias=float(error.mean()))


def fill_coordinates(frame, train):
    """Training-only lookup for the reduced December input schema."""
    result = frame.copy()
    cities = pd.concat([
        train[['pickup', 'pickup_lat', 'pickup_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
        train[['delivery', 'delivery_lat', 'delivery_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
    ]).groupby('city')[['lat', 'lon']].median()
    for side in ['pickup', 'delivery']:
        for coord in ['lat', 'lon']:
            result[f'{side}_{coord}'] = result[side].map(cities[coord])
    return result
