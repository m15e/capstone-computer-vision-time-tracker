## Udacity MLND Capstone Project
------

Building a desktop activity classifier for Mac by training a Convolutional Neural Network to distinguish between desktop screen activities using transfer learning.
The final model achieves an accuracy of 97.24% on the test set.

The project report and model results can be found at [https://github.com/m15e/capstone-computer-vision-time-tracker/blob/master/capstone_project.pdf](https://github.com/m15e/capstone-computer-vision-time-tracker/blob/master/capstone_project.pdf)

### Dependencies 
------
Please note that the first version of this application unfortunately only runs on Mac

- Python 3.5
- Pytorch
- fastai
- pyautogui
- pync

### Files
-----

All files are contained in the timeNet.

- The project report can be found under timeNet/mlnd_capstone.pdf
- The test data label mapping can be found in timeNet/test.csv
- The files for training the benchmark and final models can be found under timeNet/benchmark_model.ipynb and timeNet/classifier_training.ipynb


### Usage
------

1. Open the terminal
2. Navigate to the timenet folder
3. Run `python timeNet.py`
