# Covid-Stream Framework

## Installation
```
$ virtualenv -p python3 .venv
$ source .venv/bin/activate
$ pip3 install -r requirements.txt
```

## Phase 1 - Transfer Learning
```
$ python3 extract_features.py -d data/train -c features.csv -b 32
```

## Phase 2 - Incremental Learning
```
$ python3 train_incremental.py -c features.csv -n 100352
```