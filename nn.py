from keras.models import Model, Sequential
from keras.layers import SimpleRNN, Dense, LSTM
from tensorflow import keras


class RNN(Model):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.visualize = config.get("visualize", False)
        self.weight_file = config.get("weight_file", "")
        self.trained = False
        layers = [
            LSTM(32),
            Dense(1)
        ]
        self.model = Sequential(layers)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )
        try:
            self.load_weights(self.weight_file)
            self.trained = True
        except Exception:
            print("Unable to load weight file. Model needs to be trained.")

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        super().get_config()
