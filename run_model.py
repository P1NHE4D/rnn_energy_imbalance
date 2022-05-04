import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from yaml import safe_load

from nn import RNN
from preprocessing import preprocess_data, TimeseriesDataset


def main(config_file):
    with open(config_file) as f:
        config = safe_load(f)
    test_df = pd.read_csv(config["dataset"]["test"])
    test_df.flow = -test_df.flow

    # store mean and std to scale back predictions if enabled
    test_y_mean = test_df.y.mean()
    test_y_std = test_df.y.std()

    # preprocess data
    preprocessing_config = config.get("preprocessing", {})
    test_df = preprocess_data(test_df, **preprocessing_config)

    # construct dataset
    test = TimeseriesDataset(test_df, config["dataset"]["input_steps"])

    # setup model
    model_config = config.get("model", {})
    model = RNN(**model_config)

    # evaluate test set
    model.evaluate(x=test.x, y=test.y)

    # predict and visualize 2-hour forecast
    for _ in range(config.get("num_visualizations", 5)):

        # get random sample (ndarray<steps, features>)
        idx = np.random.choice(np.arange(0, len(test.x)))
        sample = test.x[idx].copy()

        # get gt for all steps involved for visualization purposes
        input_gt = test.input_y[idx].tolist()
        input_gt[-1] = np.nan
        forecasts = [np.nan for _ in range(len(input_gt) - 2)]
        y_true = [np.nan for _ in range(len(input_gt) - 2)]

        # append last input ground truth to forecasts and y_true for a continuous line
        forecasts.append(input_gt[-2])
        y_true.append(input_gt[-2])

        # get step size
        step_size = 5 * config["preprocessing"]["subsampling_rate"]

        # predict imbalance using given sample
        pred = model(np.array([sample]))[0][0]
        forecast_count = 1

        # append prediction and ground truth to lists used in the visualization
        forecasts.append(pred)
        y_true.append(test.y[idx])

        for _ in range(int(120 / step_size)):
            # get following sequence
            idx += 1
            sample = test.x[idx].copy()

            # replace previous_y with prediction in last step of the sample
            sample[-forecast_count:, test.indices["previous_y"]] = forecasts[-forecast_count:]

            # predict imbalance
            pred = model(np.array([sample]))[0][0]
            forecast_count += 1

            # append prediction and ground truth to lists used for visualization
            forecasts.append(pred)
            y_true.append(test.y[idx])
            input_gt.append(np.nan)

        if not config.get("display_normed_vals", True):
            input_gt = np.array(input_gt) * test_y_std + test_y_mean
            y_true = np.array(y_true) * test_y_std + test_y_mean
            forecasts = np.array(forecasts) * test_y_std + test_y_mean

        t = np.array(list(range(len(y_true))), dtype='object')
        plt.plot(t, np.array(input_gt, dtype='object'), color='#0048ff', label="hist")
        plt.plot(t, np.array(y_true, dtype='object'), color='#ff9900', label="target")
        plt.plot(t, np.array(forecasts, dtype='object'), color='#00910a', label="pred")
        plt.legend()
        plt.xlabel("Timestep")
        plt.ylabel("Normalized Imbalance")
        plt.show()


if __name__ == '__main__':
    main("config.yaml")
