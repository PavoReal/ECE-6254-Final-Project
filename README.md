# ECE 6254 Final Project

## TODO
In no particular order:
- Better prediction visualization and output
- 

## Setup

### Setup venv and install deps
Python version: 3.11.11

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Downlaod data
```bash
python src/run.py download -p ./dataset
```
This will download the dataset to the `./dataset` directory.


## Training
First create a directory to save the models within.

``` bash
mkdir -p ./models
```

To train the model with default settings, run the following command:
``` bash
python src/run.py train -m ./models/test-model -d ./dataset/2/stocks/INTC.csv
```
Training will create `./models/test-model.keras` and `./models/test-model.pkl` files.


## Testing
To test the model, run the following command, note no extension on the model name:
``` bash
python src/run.py test -m ./models/test-model
```