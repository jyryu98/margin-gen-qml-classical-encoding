# Margin-Based Generalization for QML w/ Classical Data Encoding

## Overview

Trains quantum machine learning models with classical data encoding across a grid of various hyperparameters, then tests various complexity measures' ability to predict the generalization gap using methods developed in the classical deep learning literature. 

## Environment

Create a fresh environment, then install dependencies:
```bash
pip install -r requirements.txt
```
Python 3.13 was used for the paper experiments.

## Project Structure

### Train and Score

- `experiment.py`: Helper functions for data construction, training, and testing
    - `./Fashion4v9`, `./KMNIST0v5`, `./MNIST1v6`: Contains preprocessed datasets used in the paper

- `train.py`: Performs training runs, saving results in `./Fashion4v9`, `./KMNIST0v5`, `./MNIST1v6`

- `score_and_margins.py`: Takes trained models and scores accuracies/margins

- `summarize_results.py`: Summarizes training and testing results into a csv file

- `draw_figures.ipynb`: Draws paper figures from results

### Analysis (`./analysis/`)

- `/se_test/`, `/ci_test/`, `/bootstrap/`: Contains test results shown in the paper

- `se_test.py`/`ci_test.py`: Calculates sign-errors/conditional independence tests

- `composite_optimize.py`: Optimizes composite measure inspired by the derived bound in the paper

- `extra_measures_test.py`: Calculates sign-errors/conditional independence tests for extra complexity measures guided by derived bounds

- `bootstrap_k.py`, `bootstrap_se_q90.py`: Bootstrapping for conditional independence test / sign-errors