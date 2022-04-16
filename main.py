import pandas as pd
from yaml import safe_load
from preprocessing import preprocess_data, construct_dataset
from nn import RNN


def main():
    with open("config.yaml") as f:
        config = safe_load(f)

    train_df = pd.read_csv(config["dataset"]["train"])
    test_df = pd.read_csv(config["dataset"]["test"])

    train_df = preprocess_data(config.get("preprocessing", {}), train_df)
    test_df = preprocess_data(config.get("preprocessing", {}), test_df)

    train = construct_dataset(train_df, config["dataset"]["input_steps"], config["dataset"]["batch_size"])
    test = construct_dataset(test_df, config["dataset"]["input_steps"], config["dataset"]["batch_size"])

    model_config = config.get("model", {})
    model = RNN(model_config)
    if not model.trained or model_config.get("force_retrain", False):
        model.fit(train, epochs=model_config.get("epochs", 1))
        model.save_weights(model_config.get("file_path", "models/rnn"))
    model.evaluate(test)


if __name__ == '__main__':
    main()
