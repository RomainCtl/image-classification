from evaluate import infer
from DB import Database, DB_TRAIN, DB_VALIDATION

from color import Color
from daisy import Daisy
from edge import Edge
from gabor import Gabor
from HOG import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat
from fusion import FeatureFusion

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
    """Launch a classification algorithm. Train on a first dataset, and validate it with a second (and print result)

    Args:
        method (object): Class object that have a make_samples function
        depth (int, optional): Means the system will return top-depth images. Defaults to 3.
    """
    db_train = Database(DB_TRAIN)
    db_validation = Database(DB_VALIDATION)

    train_samples = method.make_samples(db_train)
    validation_samples = method.make_samples(db_validation)

    final = {cl: [0, 0] for cl in sorted(db_train.get_class())}

    for img in validation_samples:
        selected_class = knn(img, train_samples, depth=depth)

        if img['cls'] == selected_class:
            final[img['cls']][0] += 1
        final[img['cls']][1] += 1

    # Beautiful print
    print("\n========= Result =========")
    col_width = max(len(cl) for cl, v in final.items()) + 2
    for cl, v in final.items():
        print(
            f"{cl.ljust(col_width)}: {v[0]}/{v[1]}\t{round(v[0]*100/v[1], 2)} %")


def feature(s):
    try:
        name, weight = s.split(':')
        weight = int(weight)
        if name not in features.keys() or weight < 1:
            raise Exception
        return name, weight
    except:
        raise argparse.ArgumentTypeError(
            f"\nFeature must be 'name:weight'\n\tname in {features.keys()}\n\tweight >= 1")


features = {
    "color": Color(),
    "daisy": Daisy(),
    "edge": Edge(),
    "gabor": Gabor(),
    "hog": HOG(),
    "vgg": VGGNetFeat(),
    "res": ResNetFeat(),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--depth", help="Define depth",
                        type=int, default=3)
    egroup = parser.add_mutually_exclusive_group(required=True)
    egroup.add_argument("-a", "--feature", help="Feature to launch (mutually exclusive with '--fusion')",
                        choices=list(features.keys()))
    egroup.add_argument("-f", "--fusion", help="Use feature fusion method (mutually exclusive with '--feature')",
                        type=feature, nargs="+")
    args = parser.parse_args()

    # The parser forces to have at least feature xor fusion as argument
    if args.feature:
        launch(features[args.feature], args.depth)

    if args.fusion:
        feats = {f[0]: f[1] for f in args.fusion}
        launch(FeatureFusion(features=feats), depth=args.depth)
