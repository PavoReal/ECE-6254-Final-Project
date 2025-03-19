# ECE 6254 Final Project

Phase 1 paper: Gülmez, B. (2023). Stock price prediction with optimized deep LSTM network with artificial rabbits optimization algorithm. Expert Systems With Applications, 227, 120346. `https://doi.org/10.1016/j.eswa.2023.120346`

## Setup

### Setup venv and install deps
Tested python version: 3.11.11

#### Setup venv

```bash
python -m venv venv
source venv/bin/activate
```

#### Install deps
```bash
pip install -r requirements.txt
```

### Download the dataset
You have two options for downloading data:

#### Option 1: Original Kaggle Dataset
```bash
python src/run.py download 
```

If you want to specify a custom download folder, use the `-p` flag:
```bash
python src/run.py download -p <dataset_folder>
```

#### Option 2: Fresh Data from Yahoo Finance
Download fresh data for all companies in the original dataset:
```bash
python src/run.py download_yfinance
```

Or download data for specific symbols:
```bash
python src/run.py download_yfinance --symbols SPYD AAPL GOOGL
```

You can customize the output directory and train/test split ratio:
```bash
python src/run.py download_yfinance --output ./my_fresh_data --train-ratio 0.7
```

## Adding new models
Inside `src/train.py` you'll find the `model_arch` variable. To add your model follow the pattern done for `create_model_gp` and `create_model_anu`. You'll need to give the model a name, description, and function. The function should create and return the model, but not compile it. For example:

```python3
def create_model_gp(seq_length, data_shape):
    model = Sequential()
    model.add(Input(shape=(seq_length, data_shape)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1))

    return model
```

## Usage
```
usage: run.py [-h] {train,test,compare,download,arch_list,dataset_list} ...

CLI interface for training and testing models.

positional arguments:
  {train,test,compare,download,arch_list,dataset_list}
                        Sub-command help (train or test)
    train               Train a model
    test                Test a model
    compare             Compare multiple trained models
    download            Download the dataset
    arch_list           List all available model architectures
    dataset_list        List all available datasets.

options:
  -h, --help            show this help message and exit
```

### Training a model
```
usage: run.py train [-h] -m MODEL_FILE -d DATA_NAME [--data_dir DATA_DIR]
                   [--data_source {kaggle,yfinance}] [-f FEATURES [FEATURES ...]]
                   [-a MODEL_ARCH] [-s SEQ_LENGTH] [-e EPOCHS]

options:
  -h, --help            show this help message and exit
  -m MODEL_FILE, --model_file MODEL_FILE
                        Path to the model file (without extension) to save or load the model
  -d DATA_NAME, --data_name DATA_NAME
                        Name of the dataset item, use command dataset_list for a complete list
  --data_dir DATA_DIR   Override the default dataset dir of ./dataset
  --data_source {kaggle,yfinance}
                        Choose data source: kaggle (original dataset) or yfinance (fresh data)
  -f FEATURES [FEATURES ...], --features FEATURES [FEATURES ...]
                        Features to train on
  -a MODEL_ARCH, --model_arch MODEL_ARCH
                        Change the model architecture, use command arch_list for a complete list
  -s SEQ_LENGTH, --seq_length SEQ_LENGTH
                        Sequence length for training
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
```

First create a directory to save the models within:
``` bash
mkdir -p ./models
```

To train a model using the original Kaggle dataset:
``` bash
python src/run.py train -m ./models/test-model -d SPY 
```

To train a model using fresh data from Yahoo Finance:
``` bash
python src/run.py train -m ./models/test-model -d SPY --data_source yfinance
```

Training will create `./models/test-model.keras` and `./models/test-model.pkl` files.

### Testing a model
```
usage: run.py test [-h] -m MODEL_PATH -d DATA_NAME [--data_dir DATA_DIR]
                  [--data_source {kaggle,yfinance}]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the model file (without extension)
  -d DATA_NAME, --data_name DATA_NAME
                        Name of the dataset item, use command dataset_list for a complete list
  --data_dir DATA_DIR   Override the default dataset dir of ./dataset
  --data_source {kaggle,yfinance}
                        Choose data source: kaggle (original dataset) or yfinance (fresh data)
```

To test a model using the original Kaggle dataset:
``` bash
python src/run.py test -m ./models/test-model -d SPY
```

To test a model using fresh data from Yahoo Finance:
``` bash
python src/run.py test -m ./models/test-model -d SPY --data_source yfinance
```

### Comparing multiple models
```
usage: run.py compare [-h] -m MODEL_PATHS [MODEL_PATHS ...] -d DATA_NAME
                     [--data_dir DATA_DIR] [--data_source {kaggle,yfinance}]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATHS [MODEL_PATHS ...], --model_paths MODEL_PATHS [MODEL_PATHS ...]
                        List of model file paths (without extension) to compare
  -d DATA_NAME, --data_name DATA_NAME
                        Name of dataset to use, use command dataset_list for a complete list of available datasets
  --data_dir DATA_DIR   Override the default dataset dir of ./dataset
  --data_source {kaggle,yfinance}
                        Choose data source: kaggle (original dataset) or yfinance (fresh data)
```

To compare models using the original Kaggle dataset:
```bash
python src/run.py compare -m ./models/test-model-1 ./models/test-model-2 -d SPY
```

To compare models using fresh data from Yahoo Finance:
```bash
python src/run.py compare -m ./models/test-model-1 ./models/test-model-2 -d SPY --data_source yfinance
```

To compare multiple models at once:
```bash
python src/run.py compare -m ./models/test-model-1 ./models/test-model-2 ./models/test-model-3 -d SPY
```

The comparison will generate a plot showing the actual price and predictions from all models with different colors. The plot will be saved in the `./figures` directory with a filename that includes all the model names.