import numpy as np
import pandas as pd
from scipy import interpolate


def construct_dataset(data, input_steps):
    target_df = data.pop("y")
    indices = {name: i for i, name in enumerate(data.columns)}
    input_data_target = target_df[:-1].to_numpy()
    input_data = data[:-1].to_numpy()
    targets = target_df[input_steps:].to_numpy()

    x = []
    y = []
    input_y = []
    for i in range(len(targets)):
        x.append(input_data[i:i+input_steps])
        input_y.append(input_data_target[i:i+input_steps])
        y.append(targets[i])
    return np.array(x), np.array(y), np.array(input_y), indices


def preprocess_data(config: dict, data: pd.DataFrame):
    data["date"] = pd.to_datetime(data['start_time'], format='%Y-%m-%d  %H:%M:%S').dt.date

    subsampling_rate = config.get("subsampling_rate", 1)
    if config.get("subsample", False):
        print("Subsampling...")
        data = subsample(data.copy(), subsampling_rate)
    if config.get("clamp_values", False):
        print("Clamping values...")
        data = clamp(data.copy())
    if config.get("avoid_structural_imbalance", False):
        print("Computing and subtracting structural imbalance...")
        data = avoid_structural_imbalance(data.copy())
    print("Adding selected features...")
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
        data = add_mean_imbalance(data.copy(), subsampling_rate)

    data = data.drop(["start_time", "date", "river"], axis=1)

    if config.get("normalize", False):
        print("Normalizing data...")
        data = normalize(data.copy())

    return data


def avoid_structural_imbalance(data: pd.DataFrame):
    production = data.total.to_numpy() + data.flow.to_numpy()
    x = np.arange(0, len(production))
    tck = interpolate.splrep(x, production)
    smoothed_production = interpolate.splev(x, tck)
    data.y -= smoothed_production
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


def add_mean_imbalance(data: pd.DataFrame, subsampling_rate):
    step_size = 5 * subsampling_rate
    steps_in_hour = int(60 / step_size)
    periods = steps_in_hour * 24
    data["mean_imbalance"] = data.groupby("date")["y"].transform('mean')
    data["mean_imbalance"] = data["mean_imbalance"].shift(periods)
    data.loc[:, "mean_imbalance"] = data.loc[:, "mean_imbalance"].fillna(data.y.mean())
    return data


def normalize(data: pd.DataFrame):
    return (data - data.mean()) / data.std()
