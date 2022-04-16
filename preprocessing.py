import numpy as np
import pandas as pd
from keras.preprocessing.timeseries import timeseries_dataset_from_array


def construct_dataset(data, input_steps, batch_size):
    target_df = data.pop("y")
    input_data = data[:-input_steps]
    targets = target_df[input_steps:]
    return timeseries_dataset_from_array(
        input_data,
        targets,
        sequence_length=input_steps,
        batch_size=batch_size,
    )


def preprocess_data(config: dict, data: pd.DataFrame):
    data["date"] = pd.to_datetime(data['start_time'], format='%Y-%m-%d  %H:%M:%S').dt.date

    subsampling_rate = config.get("subsampling_rate", 1)
    if config.get("subsample", False):
        data = subsample(data.copy(), subsampling_rate)
    if config.get("clamp_values", False):
        data = clamp(data.copy())
    data = add_time_features(
        data.copy(),
        config.get("time_of_day", False),
        config.get("time_of_week", False),
        config.get("time_of_year", False)
    )
    if config.get("prev_imbalance", False):
        data = add_prev_imb(data.copy())
    if config.get("imb_prev_day", False):
        data = add_imb_prev_day(data.copy(), subsampling_rate)
    if config.get("mean_imbalance", False):
        data = add_mean_imbalance(data.copy())
    if config.get("normalize", False):
        data = normalize(data.copy())

    data = data.drop(["start_time", "date", "river"], axis=1)
    return data


def subsample(data: pd.DataFrame, subsampling_rate: int):
    return data[subsampling_rate - 1::subsampling_rate]


def clamp(data: pd.DataFrame):
    l, u = data.quantile([0.005, 0.995])["y"]
    new_val = data.loc[(data.y >= l) | (data.y <= u), "y"].mean()
    data.loc[(data.y < l) | (data.y > u), "y"] = new_val
    return data


def add_time_features(
        data: pd.DataFrame,
        time_of_day=False,
        time_of_week=False,
        time_of_year=False
):
    start_time = pd.to_datetime(data['start_time'], format='%Y-%m-%d  %H:%M:%S')
    timestamp_s = start_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    week = 7 * day
    year = 365.25 * day

    if time_of_day:
        data['time_of_day'] = np.sin(timestamp_s * (2 * np.pi / day))
    if time_of_week:
        data['time_of_week'] = np.sin(timestamp_s * (2 * np.pi / week))
    if time_of_year:
        data['time_of_year'] = np.sin(timestamp_s * (2 * np.pi / year))

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


def add_mean_imbalance(data: pd.DataFrame):
    data["mean_imbalance"] = data.groupby("date")["y"].transform('mean')
    return data


def normalize(data: pd.DataFrame):
    return (data - data.mean()) / data.std()
