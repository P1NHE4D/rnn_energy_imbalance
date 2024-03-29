import numpy as np
import pandas as pd
from yaml import safe_load, safe_dump
from preprocessing import preprocess_data, TimeseriesDataset
from nn import RNN, LossHistory
import matplotlib.pyplot as plt
import argparse


def main(config_file):
    with open(config_file) as f:
        config = safe_load(f)
    if config.get("store_config", None) is not None:
        with open(config.get("store_config", None) + ".yaml", 'w') as f:
            safe_dump(config, f, default_flow_style=False)

    # load datasets
    train_df = pd.read_csv(config["dataset"]["train"])
    test_df = pd.read_csv(config["dataset"]["test"])
    train_df.flow = -train_df.flow
    test_df.flow = -test_df.flow

    # store mean and std to scale back predictions if enabled
    test_y_mean = test_df.y.mean()
    test_y_std = test_df.y.std()

    # preprocess data
    preprocessing_config = config.get("preprocessing", {})
    train_df = preprocess_data(train_df, **preprocessing_config)
    test_df = preprocess_data(test_df, **preprocessing_config)

    # construct dataset and split target from input data
    train = TimeseriesDataset(train_df, config["dataset"]["input_steps"])
    test = TimeseriesDataset(test_df, config["dataset"]["input_steps"])

    # train model
    model_config = config.get("model", {})
    model = RNN(**model_config)
    if not model.trained or model_config.get("force_retrain", False):
        history = LossHistory()
        model.fit(x=train.x, y=train.y, validation_data=(test.x, test.y), epochs=model_config.get("epochs", 1),
                  batch_size=config["dataset"]["batch_size"], callbacks=[history])
        model.save_weights(model_config.get("file_path", "models/rnn"))
        if model_config.get("visualize", False):
            plt.plot(np.arange(1, history.epochs + 1), history.losses, color='#0048ff', label="Train")
            if len(history.val_losses) != 0:
                plt.plot(np.arange(1, history.epochs + 1), history.val_losses, color='#ff0000', label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("MSE")
            plt.legend()
            plt.show()

    # evaluate model on the given test set
    model.evaluate(x=test.x, y=test.y)

    # visualization
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
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file')
    args = parser.parse_args()
    main(args.config)
