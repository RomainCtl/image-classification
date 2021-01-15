# Image Classification

## Dependencies

[The COREL Database](https://sites.google.com/site/dctresearch/Home/content-based-image-retrieval): The database is divided into 7 archives, I downloaded all of them, and merged them with the command: `cat CorelDB.7z.00* > CorelDB.7z`. You can find this file in `./data/CorelDB.7z`.

Install the dependencies:
```
make install
```

## Usage

Random split coreldb to 3 dataset (test, train, validation):
```
make split-dataset
```

### Image classification based on attributes (*CBIR* = *content-based image retrieval*)

Feature list: `[color, daisy, edge, gabor, hog, vgg, res]`

```bash
# Launch classification using color feature
make cbir -- -a color

# Launch feature fusion between daisy and color (with weight 2 and 5)
make cbir -- -f daisy:2 color:5
```

If you do not want to use `make`:
```bash
# Launch classification using color feature
cd src/CBIR && python -m pipenv run python scripts/classify.py -a color

# Launch feature fusion between daisy and color (with weight 2 and 5)
cd src/CBIR && python -m pipenv run python scripts/classify.py -f daisy:2 color:5
```

Full usage help (from `make cbir -- --help`):
```
usage: classify.py [-h] [-D DEPTH] (-a {color,daisy,edge,gabor,hog,vgg,res} | -f FUSION [FUSION ...])

optional arguments:
  -h, --help            show this help message and exit
  -D DEPTH, --depth DEPTH
                        Define depth
  -a {color,daisy,edge,gabor,hog,vgg,res}, --feature {color,daisy,edge,gabor,hog,vgg,res}
                        Feature to launch
  -f FUSION [FUSION ...], --fusion FUSION [FUSION ...]
                        Use feature fusion method
```

### Image classification based on a neural network (*CNN* = *convolutional neural network*)

```bash
# Launch CNN classification with default args
make cnn

# Launch CNN classification with epochs=50 and rewrite weights (re-train the model)
make cnn -- -e 50 -R
```

If you do not want to use `make`:
```bash
# Launch CNN classification with default args
cd src/CNN && python -m pipenv run python neural_model.py

# Launch CNN classification with epochs=50 and rewrite weights (re-train the model)
cd src/CNN && python -m pipenv run python neural_model.py -e 50 -R
```

Full usage help (from `make cnn -- --help`):
```
usage: neural_model.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-R] [-H]

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -R, --rewrite-weights
                        Re-write weights
  -H, --hide-plots      Hide plots
```
