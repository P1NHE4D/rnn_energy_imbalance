import numpy as np
import pandas as pd
from yaml import safe_load
from preprocessing import preprocess_data, construct_dataset
from nn import RNN, LossHistory
import matplotlib.pyplot as plt


def main():
    with open("config.yaml") as f:
        config = safe_load(f)

    train_df = pd.read_csv(config["dataset"]["train"])
    test_df = pd.read_csv(config["dataset"]["test"])

    train_df = preprocess_data(config.get("preprocessing", {}), train_df)
    test_df = preprocess_data(config.get("preprocessing", {}), test_df)

    x_train, y_train, _ = construct_dataset(train_df, config["dataset"]["input_steps"])
    x_test, y_test, input_y_test = construct_dataset(test_df, config["dataset"]["input_steps"])

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
    model.evaluate(x=x_test, y=y_test)

    # visualization
    idx = np.random.choice(np.arange(0, len(x_test)))
    sample = x_test[idx]
    input_gt = input_y_test[idx].tolist()
    forecasts = [np.nan for _ in range(len(input_gt) - 1)]
    y_true = [np.nan for _ in range(len(input_gt) - 1)]
    forecasts.append(input_gt[-1])
    y_true.append(input_gt[-1])
    step_size = 5 * config["preprocessing"]["subsampling_rate"]

    pred = model(np.array([sample]))[0][0]
    forecasts.append(pred)
    y_true.append(y_test[idx])
    input_gt.append(np.nan)
    for _ in range(int(120 / step_size)):
        idx += 1
        sample = x_test[idx]
        sample[5] = pred
        pred = model(np.array([sample]))[0][0]
        forecasts.append(pred)
        y_true.append(y_test[idx])
        input_gt.append(np.nan)
    t = list(range(len(y_true)))
    plt.plot(t, input_gt, color='#0048ff', label="hist")
    plt.plot(t, y_true, color='#ff9900', label="target")
    plt.plot(t, forecasts, color='#00910a', label="pred")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Normalized Imbalance")
    plt.show()


if __name__ == '__main__':
    main()
