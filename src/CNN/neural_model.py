from tensorflow import argmax
from tensorflow.keras import applications, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os
import argparse

from database import Database


class CNNModel(Sequential):
    def __init__(self, database, show_plot=False, rewrite_weights=False):
        super().__init__()

        self.database = database
        self.show_plot = show_plot
        self.rewrite_weights = rewrite_weights

        self.target_size = (150, 150)

        self.add(Conv2D(32, (3, 3), input_shape=(*self.target_size, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(32, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(64, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)
        self.add(Flatten())
        self.add(Dense(64))
        self.add(Activation('relu'))
        self.add(Dropout(0.5))
        self.add(Dense(len(self.database)))
        self.add(Activation('softmax'))

        self.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, batch_size=16, epochs=50):
        # if weights exist, just load previously generated model
        weights_path = os.path.join(
            'result', self.database.get_weights_filename())
        if os.path.isfile(weights_path) and not self.rewrite_weights:
            self.load_weights(weights_path)
            print("Model already performed !")
            return

        train = self.database.get_image_data_generator('train')
        validation = self.database.get_image_data_generator('validation')

        # Model training
        history = self.fit(train, epochs=epochs, validation_data=validation)

        # Save weights (in a HDF5 file)
        self.save_weights(weights_path)

        if self.show_plot:
            # Show plot
            for key in history.history.keys():
                plt.plot(history.history[key])

            plt.ylabel("accuracy or loss")
            plt.xlabel("epoch")
            plt.legend(history.history.keys())
            plt.show()

    def test(self):
        test = self.database.get_image_data_generator("test", shuffle=False)

        # Predict
        predictions = argmax(self.predict(test), axis=1).numpy()

        if self.show_plot:
            cmatrix = confusion_matrix(
                test.classes, predictions, normalize='true')
            display = ConfusionMatrixDisplay(
                confusion_matrix=cmatrix, display_labels=self.database.get_class())
            display.plot(cmap="Greens")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-R", "--rewrite-weights", help="Re-write weights",
                        action="store_true")
    parser.add_argument("-H", "--hide-plots", help="Hide plots",
                        action="store_true")
    args = parser.parse_args()

    #
    model = CNNModel(
        database=Database(),
        rewrite_weights=args.rewrite_weights,
        show_plot=(not args.hide_plots)
    )
    model.train(batch_size=args.batch_size, epochs=args.epochs)
    model.test()
