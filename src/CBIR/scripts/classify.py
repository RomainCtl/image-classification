from evaluate import infer
from color import Color
from daisy import Daisy
from DB import Database, DB_TRAIN, DB_VALIDATION


from itertools import groupby
from statistics import mean
import argparse


def knn(query, samples, depth=3):
    """Infer a query with K-nearsest neighbors strategy

    Args:
        query (dict): { 'img': <path_to_img>, 'cls': <img class>, 'hist' <img histogram> }
        samples (list): list of { 'img': <path_to_img>, 'cls': <img class>, 'hist' <img histogram> }
        depth (int, optional): retrieved depth during inference. Defaults to 3.

    Returns:
        str: selected class
    """
    _, results = infer(query, samples, depth=depth)
    # Group by class
    res = sorted(results, key=lambda x: x['cls'])
    grouped_by_class = {
        k: list(map(lambda x: x['dis'], g))
        for k, g in groupby(res, key=lambda x: x['cls'])
    }
    # Calc average by class
    grouped_by_class = {
        k: mean(v) for k, v in grouped_by_class.items()
        # k: len(v) for k, v in grouped_by_class.items()
    }
    # Get max
    selected_class = sorted(
        grouped_by_class.items(), key=lambda x: x[1]
    )[:1][0][0]

    return selected_class


def launch(method, depth: int = 3):
    db_train = Database(DB_TRAIN)
    db_validation = Database(DB_VALIDATION)

    train_samples = method.make_samples(db_train)
    validation_samples = method.make_samples(db_validation)

    final = {cl: [0, 0] for cl in db_train.get_class()}

    for img in validation_samples:
        selected_class = knn(img, train_samples, depth=depth)

        if img['cls'] == selected_class:
            final[img['cls']][0] += 1
        final[img['cls']][1] += 1

    # Beautiful print
    col_width = max(len(cl) for cl, v in final.items()) + 2
    for cl, v in final.items():
        print(
            f"{cl.ljust(col_width)}: {v[0]}/{v[1]}\t{round(v[0]*100/v[1], 2)} %")


if __name__ == "__main__":
    algo = {
        "color": Color(),
        "daisy": Daisy()
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--depth", help="Define depth",
                        type=int, default=3)
    parser.add_argument("-a", "--algo", help="Algorithm to launch",
                        choices=list(algo.keys()))
    args = parser.parse_args()

    if args.algo:
        launch(algo[args.algo], args.depth)
