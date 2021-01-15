from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# DB_TRAIN = '../../data/train'
# DB_VALIDATION = '../../data/validation'
# DB_TEST = '../../data/test'
DB_PATH = '../../data/'


class Database(object):

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def __len__(self):
        return len(self.get_class())

    def get_class(self):
        """Get list of class in database

        Returns:
            list: sorted list of class in database
        """
        return sorted(os.listdir(os.path.join(self.db_path, 'train')))

    def get_weights_filename(self):
        """Get Weights filename (weights generated after training or during training)

        Returns:
            str: the filename.
        """
        return '-'.join(self.get_class())+'.h5'

    def get_image_data_generator(self, subfolder, shuffle=True, target_size=(150, 150), batch_size=16):
        """Get a generator of tensor image data with real-time data augmentation

        Args:
            subfolder (str): Subfolder of the database (train, validation, test).
            shuffle (bool): Shuffle or in order (data). Default to True.
            target_size (tuple, optional): Tuple (with, height) used to resize images. Defaults to (150, 150).
            batch_size (int, optional): Image per chunk. Defaults to 16.

        Returns:
            DirectoryIterator: Yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size) and y is anumpy array of cooresponding labels.
        """
        return ImageDataGenerator(
            rescale=1./255
        ).flow_from_directory(
            os.path.join(self.db_path, subfolder),
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=shuffle
        )


if __name__ == "__main__":
    db = Database()
    classes = db.get_class()

    print("DB length:", len(db))
    print(classes)
    print(db.get_weights_filename())
