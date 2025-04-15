#python .\src\run.py train -m .\models_kaggle\ann       -a ann       -d SPY --data_source kaggle
#python .\src\run.py train -m .\models_kaggle\lstm1d    -a lstm1d    -d SPY --data_source kaggle
#python .\src\run.py train -m .\models_kaggle\lstm2d    -a lstm2d    -d SPY --data_source kaggle
#python .\src\run.py train -m .\models_kaggle\lstm3d    -a lstm3d    -d SPY --data_source kaggle
#python .\src\run.py train -m .\models_kaggle\lstm-garo -a lstm-garo -d SPY -t 3300 --data_source kaggle
#python .\src\run.py train -m .\models_kaggle\lstm-garo-large -a lstm-garo-large -d SPY -t 1750 --data_source kaggle
#python .\src\run.py train -m .\models_kaggle\randForest -a randForest -d SPY -l 5 --data_source kaggle

#python .\src\run.py compare -m .\models_kaggle\ann .\models_kaggle\lstm-garo-large .\models_kaggle\randForest .\models_kaggle\lstm-garo .\models_kaggle\lstm1d .\models_kaggle\lstm2d .\models_kaggle\lstm3d -d SPY --data_source kaggle
#python .\src\run.py compare -m .\models_kaggle\ann .\models_kaggle\lstm-garo-large .\models_kaggle\randForest .\models_kaggle\lstm-garo .\models_kaggle\lstm1d .\models_kaggle\lstm2d .\models_kaggle\lstm3d -d SPY --data_source yfinance
# python .\src\run.py compare -m .\models\ann .\models\lstm-garo-large .\models\randForest  .\models\lstm-garo .\models\lstm1d .\models\lstm2d .\models\lstm3d -d SPY --data_source kaggle
# python .\src\run.py compare -m .\models\ann .\models\lstm-garo-large .\models\randForest  .\models\lstm-garo .\models\lstm1d .\models\lstm2d .\models\lstm3d -d SPY --data_source yfinance

# same as above but for WSL/Linux
python3 ./src/run.py train -m ./models/ann -a ann -d SPY --data_source kaggle
python3 ./src/run.py train -m ./models/lstm1d -a lstm1d -d SPY --data_source kaggle
python3 ./src/run.py train -m ./models/lstm2d -a lstm2d -d SPY --data_source kaggle
python3 ./src/run.py train -m ./models/lstm3d -a lstm3d -d SPY --data_source kaggle
python3 ./src/run.py train -m ./models/lstm-garo -a lstm-garo -d SPY -t 3300 --data_source kaggle
python3 ./src/run.py train -m ./models/lstm-garo-large -a lstm-garo-large -d SPY -t 1750 --data_source kaggle
python3 ./src/run.py train -m ./models/randForest -a randForest -d SPY -l 5 --data_source kaggle

python3 ./src/run.py compare -m ./models/ann ./models/lstm-garo-large ./models/randForest ./models/lstm-garo ./models/lstm1d ./models/lstm2d ./models/lstm3d -d SPY --data_source kaggle
#python ./src/run.py compare -m ./models/ann ./models/lstm-garo-large ./models/randForest ./models/lstm-garo ./models/lstm1d ./models/lstm2d ./models/lstm3d -d SPY --data_source yfinance