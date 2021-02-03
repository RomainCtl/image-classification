from tensorflow import argmax
from tensorflow.keras import applications, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os
import shutil
import argparse

from database import Database, RESULT_RETRIEVAL

# Usefull links:
# http://playground.tensorflow.org/ Understand and test a neural network (tensorflow)
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html (this code is inspired from this page)


class CNNModel(Sequential):
    def __init__(self, database, show_plot=False, rewrite_weights=False, store_test_result=False):
        super().__init__()

        self.database = database
        self.show_plot = show_plot
        self.rewrite_weights = rewrite_weights
        self.store_test_result = store_test_result

        # # Create a simple stack of 3 convolution layer with ReLU (Rectified Linear Unit) activation and followed by max-pooling layers
        # 150, 150 = target_size (width, height)
        self.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(32, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(64, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # # The model so far outputs 3D feature maps (height, width, features)
        # Flatten: this converts our 3D feature maps to 1D feature vectors (To end the model with a single unit)
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
        """Train the model (use train and validation dataset)

        Args:
            batch_size (int, optional): Size of the batchs. Defaults to 16.
            epochs (int, optional): Number of learning cycles. Defaults to 50.
        """
        # if weights exist, just load previously generated model
        weights_path = os.path.join(
            'result', self.database.get_weights_filename())
        if os.path.isfile(weights_path) and not self.rewrite_weights:
            self.load_weights(weights_path)
            print("Model already performed !")
            return

        train = self.database.get_image_data_generator(
            'train', batch_size=batch_size)
        validation = self.database.get_image_data_generator(
            'validation', batch_size=batch_size)

        # Model training
        history = self.fit(train, epochs=epochs, validation_data=validation)

        # Save weights (in a HDF5 file)
        self.save_weights(weights_path)

        if self.show_plot:
            # Show plot
            for key in history.history.keys():
                plt.plot(history.history[key])

            # legend start with 'val_*' if for the validation dataset
            plt.ylabel("accuracy or loss")
            plt.xlabel("epoch")
            plt.legend(history.history.keys())
            plt.show()

    def test(self):
        """Test the generated model with the 'test' dataset. This function will predict class of images in 'test' dataset.
        It will display a confusion matrix if 'self.show_plot' is True (using matplotlib).
        """
        test = self.database.get_image_data_generator("test", shuffle=False)

        # Predict
        predictions = argmax(self.predict(test), axis=1).numpy()

        # Store image class result if asked
        if self.store_test_result:
            # Delete previous result
            for classe in self.database.get_class():
                shutil.rmtree(os.path.join(RESULT_RETRIEVAL, classe), ignore_errors=True)
            # and re-create directory
            for classe in self.database.get_class():
                os.mkdir(os.path.join(RESULT_RETRIEVAL, classe))

            for i in range(len(predictions)):
                selected_class = self.database.get_class()[predictions[i]]
                filename = os.path.basename(test.filenames[i])

                shutil.copy(
                    os.path.join(self.database.db_path, "test", test.filenames[i]),
                    os.path.join(RESULT_RETRIEVAL, selected_class, filename)
                )

        if self.show_plot:
            # show result plot
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
    parser.add_argument("-s", help="Store test result (in src/CNN/result/retrieval/)",
                        action="store_true")
    args = parser.parse_args()

    #
    model = CNNModel(
        database=Database(),
        rewrite_weights=args.rewrite_weights,
        show_plot=(not args.hide_plots),
        store_test_result=args.s
    )
    model.train(batch_size=args.batch_size, epochs=args.epochs)
    model.test()
