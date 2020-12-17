# Image Classification

## Dependencies

[The COREL Database](https://sites.google.com/site/dctresearch/Home/content-based-image-retrieval): The database is divided into 7 archives, I downloaded all of them, and merged them with the command: `cat CorelDB.7z.00* > CorelDB.7z`. You can find this file in `./data/CorelDB.7z`.

Install the dependencies:
```
make install
```

## Usage

Split coreldb to 3 dataset (test, train, validation):
```
make split-dataset
```

*coming-soon*
