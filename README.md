# Margin-Based Generalization for QML w/ Classical Data Encoding

## Overview

Trains quantum machine learning models with classical data encoding across a grid of various hyperparameters, then tests various complexity measures' ability to predict the generalization gap using methods developed in the classical deep learning literature. 

## Project Structure

- `experiment.py`: Helper functions for data construction, training, and testing
    - `./Fashion4v9`, `./KMNIST0v5`, `./MNIST1v6`: Contains preprocessed datasets used in the paper

- `train.py`: Performs training runs, saving results in `./Fashion4v9`, `./KMNIST0v5`, `./MNIST1v6`

- `score_and_margins.py`: Takes trained models and calculates test accuracies/margins

- `summarize_results.py`: Summarizes training and testing results into a csv file

- `se_test.py`/`ci_test.py`: Calculates sign-errors/conditional independence tests for the complexity measures considered in the paper
    - `./se_test`, `./ci_test`: Contains test results shown in the paper

- `composite_optimize.py`: Optimizes composite measure inspired by the derived bound in the paper
    - `./se_test`: Contains optimization results reported in the paper

- `draw_figures.ipynb`: Draws paper figures from results