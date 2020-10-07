# Kaggle Landmark Recognition 2020 competition: Winner solution

This repository contains the code for our winning solution to the 2020 edition of the Google Landmark Recognition competition hosted on Kaggle: https://www.kaggle.com/c/landmark-recognition-2020/leaderboard

The full solution is described in a paper hosted on arxiv: https://arxiv.org/abs/2010.01650

In order to run this code you need the train and test data from GLDv2: https://github.com/cvdfoundation/google-landmark

To train a model, please run ```src/train.py``` with a config file as flag:
```
python train.py --config config1
```

You need to adjust data paths and other parameters in respective config file to make it work.

The blending and ranking procedure is detailed in ```notebooks/blend_ranking.ipynb```.

