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

### Image classification based on attributes (*CBIR*)

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

### Image classification based on a neural network

*coming soon*
