import numpy as np
import pandas as pd
from yaml import safe_load
from preprocessing import preprocess_data, construct_dataset
from nn import RNN, LossHistory
import matplotlib.pyplot as plt
import argparse


def main(config_file):
    with open(config_file) as f:
        config = safe_load(f)

    # load datasets
    train_df = pd.read_csv(config["dataset"]["train"])
    test_df = pd.read_csv(config["dataset"]["test"])
    train_df.flow = -train_df.flow
    test_df.flow = -test_df.flow

    # store mean and std to scale back predictions if enabled
    test_y_mean = test_df.y.mean()
    test_y_std = test_df.y.std()

    # preprocess data
    train_df = preprocess_data(config.get("preprocessing", {}), train_df)
    test_df = preprocess_data(config.get("preprocessing", {}), test_df)

    # construct dataset and split target from input data
    x_train, y_train, input_y_train, train_indices = construct_dataset(train_df, config["dataset"]["input_steps"])
    x_test, y_test, input_y_test, test_indices = construct_dataset(test_df, config["dataset"]["input_steps"])

    # train model
    model_config = config.get("model", {})
    model = RNN(model_config)
    if not model.trained or model_config.get("force_retrain", False):
        history = LossHistory()
        model.fit(x=x_train, y=y_train, epochs=model_config.get("epochs", 1), batch_size=config["dataset"]["batch_size"], callbacks=[history])
        model.save_weights(model_config.get("file_path", "models/rnn"))
        if model_config.get("visualize", False):
            plt.plot(np.arange(1, history.epochs + 1), history.losses, color='#0048ff')
            plt.xlabel("Epochs")
            plt.ylabel("MSE")
            plt.show()

    # evaluate model on the given test set
    model.evaluate(x=x_test, y=y_test)

    # visualization
    for _ in range(config.get("num_visualizations", 5)):

        # get random sample (ndarray<steps, features>)
        idx = np.random.choice(np.arange(0, len(x_test)))
        sample = x_test[idx]

        # get gt for all steps involved for visualization purposes
        input_gt = input_y_test[idx].tolist()
        forecasts = [np.nan for _ in range(len(input_gt) - 1)]
        y_true = [np.nan for _ in range(len(input_gt) - 1)]

        # append last input ground truth to forecasts and y_true for a continuous line
        forecasts.append(input_gt[-1])
        y_true.append(input_gt[-1])

        # get step size
        step_size = 5 * config["preprocessing"]["subsampling_rate"]

        # predict imbalance using given sample
        pred = model(np.array([sample]))[0][0]

        # append prediction and ground truth to lists used in the visualization
        forecasts.append(pred)
        y_true.append(y_test[idx])
        input_gt.append(np.nan)

        for _ in range(int(120 / step_size)):
            # get following sequence
            idx += 1
            sample = x_test[idx]

            # replace previous_y with prediction in last step of the sample
            sample[-1, test_indices["previous_y"]] = pred

            # predict imbalance
            pred = model(np.array([sample]))[0][0]

            # append prediction and ground truth to lists used for visualization
            forecasts.append(pred)
            y_true.append(y_test[idx])
            input_gt.append(np.nan)

        if not config.get("display_normed_vals", True):
            input_gt = np.array(input_gt) * test_y_std + test_y_mean
            y_true = np.array(y_true) * test_y_std + test_y_mean
            forecasts = np.array(forecasts) * test_y_std + test_y_mean

        t = list(range(len(y_true)))
        plt.plot(t, input_gt, color='#0048ff', label="hist")
        plt.plot(t, y_true, color='#ff9900', label="target")
        plt.plot(t, forecasts, color='#00910a', label="pred")
        plt.legend()
        plt.xlabel("Timestep")
        plt.ylabel("Normalized Imbalance")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file')
    args = parser.parse_args()
    main(args.config)
