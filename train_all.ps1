python .\src\run.py train -m .\models\ann       -a ann       -d SPY --data_source kaggle
python .\src\run.py train -m .\models\lstm1d    -a lstm1d    -d SPY --data_source kaggle
python .\src\run.py train -m .\models\lstm2d    -a lstm2d    -d SPY --data_source kaggle
python .\src\run.py train -m .\models\lstm3d    -a lstm3d    -d SPY --data_source kaggle
python .\src\run.py train -m .\models\lstm-garo -a lstm-garo -d SPY -t 4 --data_source kaggle

python .\src\run.py compare -m .\models\ann .\models\lstm-garo .\models\lstm1d .\models\lstm2d .\models\lstm3d -d SPY --data_source kaggle
python .\src\run.py compare -m .\models\ann .\models\lstm-garo .\models\lstm1d .\models\lstm2d .\models\lstm3d -d SPY --data_source yfinance

