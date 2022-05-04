from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import interpolate


class TimeseriesDataset:

    def __init__(self, data, input_steps):
        x, y, input_y, indices = construct_dataset(data, input_steps)
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.input_y: np.ndarray = input_y
        self.indices: dict = indices


def construct_dataset(data, input_steps):
    target_df = data.pop("y")
    indices = {name: i for i, name in enumerate(data.columns)}
    input_data_target = target_df.to_numpy()
    input_data = data.to_numpy()
    targets = target_df[input_steps - 1:].to_numpy()

    x = []
    y = []
    input_y = []
    for i in range(len(targets)):
        x.append(input_data[i:i + input_steps])
        input_y.append(input_data_target[i:i + input_steps])
        y.append(targets[i])
    return np.array(x), np.array(y), np.array(input_y), indices


def preprocess_data(
        data: pd.DataFrame,
        subsampling_rate=1,
        clamp_values=False,
        avoid_structural_imbalance=False,
        features=None,
        normalize=False,
        **kwargs
):
    if features is None:
        features = {}

    data["date"] = pd.to_datetime(data['start_time'], format='%Y-%m-%d  %H:%M:%S').dt.date

    if subsampling_rate > 1:
        print("Subsampling...")
        data = subsample(data.copy(), subsampling_rate)
    if clamp_values:
        print("Clamping values...")
        data = clamp(data.copy())
    if avoid_structural_imbalance:
        print("Computing and subtracting structural imbalance...")
        data = subtract_structural_imbalance(data.copy())
    print("Adding selected features...")
    data = add_time_features(data.copy())
    data = add_prev_imb(data.copy())
    data = add_imb_prev_day(data.copy(), subsampling_rate)
    data = add_mean_imbalance(data.copy(), subsampling_rate)
    drop_features = [feat for feat, enabled in features.items() if not enabled]
    print("Dropping: {}".format(drop_features))

    data = data.drop(["start_time", "date"], axis=1)
    data = data.drop(drop_features, axis=1)

    if normalize:
        print("Normalizing data...")
        data = normalize_data(data.copy())

    return data


def subtract_structural_imbalance(data: pd.DataFrame):
    production = data.total.to_numpy() + data.flow.to_numpy()
    x = np.arange(0, len(production))
    tck = interpolate.splrep(x, production)
    smoothed_production = interpolate.splev(x, tck)
    structural_imbalance = production - smoothed_production
    data.y -= structural_imbalance
    return data


def subsample(data: pd.DataFrame, subsampling_rate: int):
    return data[subsampling_rate - 1::subsampling_rate]


def clamp(data: pd.DataFrame):
    l, u = data.quantile([0.005, 0.995])["y"]
    data.loc[(data.y < l), "y"] = l
    data.loc[(data.y > u), "y"] = u
    return data


def add_time_features(
        data: pd.DataFrame,
):
    start_time = pd.to_datetime(data['start_time'], format='%Y-%m-%d  %H:%M:%S')
    timestamp_s = start_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    week = 7 * day
    year = 365.25 * day

    data['time_of_day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data['time_of_day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    data['time_of_week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    data['time_of_week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    data['time_of_year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    data['time_of_year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return data


def add_prev_imb(data: pd.DataFrame):
    data["previous_y"] = data["y"].shift(1)
    data = data[1:]
    return data


def add_imb_prev_day(data: pd.DataFrame, subsampling_rate):
    step_size = 5 * subsampling_rate
    steps_in_hour = int(60 / step_size)
    periods = steps_in_hour * 24
    data["imb_prev_day"] = data["y"].shift(periods)
    data.loc[:, "imb_prev_day"] = data.loc[:, "imb_prev_day"].fillna(data.y.mean())
    return data


def add_mean_imbalance(data: pd.DataFrame, subsampling_rate):
    step_size = 5 * subsampling_rate
    steps_in_hour = int(60 / step_size)
    periods = steps_in_hour * 24
    data["mean_imbalance"] = data.groupby("date")["y"].transform('mean')
    data["mean_imbalance"] = data["mean_imbalance"].shift(periods)
    data.loc[:, "mean_imbalance"] = data.loc[:, "mean_imbalance"].fillna(data.y.mean())
    return data


def normalize_data(data: pd.DataFrame):
    return (data - data.mean()) / data.std()
