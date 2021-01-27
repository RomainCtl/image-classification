import os
import stat
import shutil
import argparse
from random import shuffle


def choose_classes(number: int, dirname: str):
    """Randomly choose n classes from coreldb

    Args:
        number (int): number of selected classes
        dirname (str): dirname where we can find the classes

    Returns:
        list: list of dir (classes) name
    """
    files_list = os.listdir(dirname)

    classes_list = list(
        filter(
            lambda file: os.path.isdir(os.path.join(dirname, file)),
            files_list
        )
    )

    shuffle(classes_list)

    return classes_list[:number]


def split_data(classes: list, data_path: str, coreldb_path: str, test: float = 0.7, train: float = 0.24, validation: float = 0.06):
    """Divide each classes of images into 3 sub-directories

    Args:
        classes (list): list of selected classes (list of string)
        test (float): proportion of test data (between 0 and 1)
        train (float): proportion of train data (between 0 and 1)
        validation (float): proportion of validation data (between 0 and 1)
    """
    assert train + validation + test == 1

    folders = ('test', 'train', 'validation')
    # Delete old datasets
    for folder in folders:
        shutil.rmtree(os.path.join(data_path, folder), ignore_errors=True)

    # Create news one
    for folder in folders:
        os.mkdir(os.path.join(data_path, folder))

    # Copy and split all image in the three folders
    for classe in classes:
        img_list = os.listdir(os.path.join(coreldb_path, classe))

        shuffle(img_list)

        indexes = (
            0,
            int(test*len(img_list)),
            int((test+train)*len(img_list)),
            len(img_list)
        )

        for k in range(3):
            # create dir
            os.mkdir(os.path.join(data_path, folders[k], classe))

            # move img
            for img in img_list[indexes[k]: indexes[k+1]]:
                shutil.copy(
                    os.path.join(coreldb_path, classe, img),
                    os.path.join(data_path, folders[k], classe, img)
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--coreldb-path",
        type=str,
        default="data/CorelDB/",
        help="Corel db path (default: data/CorelDB)")
    parser.add_argument(
        "-d", "--data-path",
        type=str,
        default="data/",
        help="Data path (default: data/)")
    parser.add_argument(
        "-C", "--classes",
        nargs="+",
        default=None,
        help="Selected classes (default: random)")
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of classes if '--classes' not used (default: 5)")
    args = parser.parse_args()

    selected_class = args.classes
    if selected_class is None:
        selected_class = choose_classes(args.n, args.coreldb_path)

    split_data(
        selected_class,
        args.data_path,
        args.coreldb_path
    )

    print("Selected class: ", ", ".join(selected_class))
