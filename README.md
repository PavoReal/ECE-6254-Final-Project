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
```
usage: run.py download [-h] [-p PATH]

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to save the dataset, defaults to ./dataset
```
To download the dataset into the default `./dataset` folder:

```bash
python src/run.py download 
```

If you want to specify a custom download folder, use the `-p` flag:
```bash
python src/run.py download -p <dataset_folder>
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
usage: run.py train [-h] -m MODEL_FILE -d DATA_NAME [--data_dir DATA_DIR] [-f FEATURES [FEATURES ...]] [-a MODEL_ARCH] [-s SEQ_LENGTH] [-e EPOCHS]

options:
  -h, --help            show this help message and exit
  -m MODEL_FILE, --model_file MODEL_FILE
                        Path to the model file (without extension) to save or load the model
  -d DATA_NAME, --data_name DATA_NAME
                        Name of the dataset item, use command dataset_list for a complete list
  --data_dir DATA_DIR   Override the default dataset dir of ./dataset
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

Next, to train the model with default settings, run the following command:
``` bash
python src/run.py train -m ./models/test-model -d SPY 
```

Training will create `./models/test-model.keras` and `./models/test-model.pkl` files.

### Testing a model
```
usage: run.py test [-h] -m MODEL_PATH -d DATA_NAME [--data_dir DATA_DIR]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the model file (without extension)
  -d DATA_NAME, --data_name DATA_NAME
                        Name of the dataset item, use command dataset_list for a complete list
  --data_dir DATA_DIR   Override the default dataset dir of ./dataset
```

Basic usage:
``` bash
python src/run.py test -m ./models/test-model -d SPY
```

### Comparing multiple models
```
usage: run.py compare [-h] -m MODEL_PATHS [MODEL_PATHS ...] -d DATA_NAME [--data_dir DATA_DIR]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATHS [MODEL_PATHS ...], --model_paths MODEL_PATHS [MODEL_PATHS ...]
                        List of model file paths (without extension) to compare
  -d DATA_NAME, --data_name DATA_NAME
                        Name of dataset to use, use command dataset_list for a complete list of available datasets
  --data_dir DATA_DIR   Override the default dataset dir of ./dataset
```

Basic usage to compare two models:
```bash
python src/run.py compare -m ./models/test-model-1 ./models/test-model-2 -d SPY
```

To compare multiple models at once:
```bash
python src/run.py compare -m ./models/lstm1d-model ./models/lstm2d-model ./models/lstm3d-model -d SPY
```

The comparison will generate a plot showing the actual price and predictions from all models with different colors. The plot will be saved in the `./figures` directory with a filename that includes all the model names.