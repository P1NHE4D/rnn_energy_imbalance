from keras.callbacks import Callback
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import L1, L2
from tensorflow import keras


class LossHistory(Callback):

    def __init__(self):
        super().__init__()
        self.losses = []
        self.epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.epochs += 1


class RNN(Model):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.dropout = config.get("dropout", 0)
        self.weight_file = config.get("weight_file", "")
        self.trained = False
        reg_rate = config.get("reg_rate", 0.01)
        reg_type = config.get("regularization", None)
        reg = L1(reg_rate) if reg_type == "l1" else L2(reg_rate) if reg_type == "l2" else None
        layers = [
            LSTM(32, dropout=self.dropout, kernel_regularizer=reg),
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
