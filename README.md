# Data Challenge Problem

The following assingment is meant to assess your abilities as a data engineer. You will be asked to modify code to better populate missing data.

## Overview
Imputing and/or modeling missing data is a task that often appears in many data-driven projects. This challenge begins with ./data/raw_diamond_data.csv, an open source data file about diamonds that has been modified to have approximately 20% of the pricing information removed. The goal of this challenge is for you to iterate on ways to better fill in the missing diamond data.

## Running the code

### Prerequisites

You will need some python packages which are listed in `requirements.txt`. It is best
to use a virtual environment. For example:

`python -m virtualenv venv`

`source venv/bin/activate` (on Mac/Linux)
OR
`venv\Scripts\activate` (on Windows)

`pip install -r requirements.txt`

When you are done using the virtual environment, run the `deactivate` command.

### Sample code

At the root directory, enter:

`python .\assignment.py`

The code reads in the data in ./data/raw_diamond_data.csv, removes all non-numerical data from the dataset, and runs a k-nearest-neighbor imputation to populate the missing price data. Next, the code reads in the complete data set from ./data/complete_diamond_data.csv and the generated prices are compared to the known real prices. The default code produces an R-squared value of around 0.86. A perfect model would produce an R-squared value of 1.0. 

## Assignment

You are tasked to perform three objectives to succesfully complete the assignment. Take notes on how each object affects the performance metric and be prepared to discuss the reasons why.

### Objective 1: Implement One-Hot Encoding
Complete the function `convert_to_one_hot_encoding` so the textual data can be used to help predict the price instead of being ignored. This method should use the sklearn.preprocessing.OneHotEncoder` class and accept and return the data types denoted in the function type hints.

### Objective 2: Implement Normalization
Complete the functions `normalize_data` and `rescale_data` so the numerical data will be normalized between [0, 1] for imputation, and rescaled afterwards. This method should use the sklearn.preprocessing.MinMaxScaler` class and accept and return the data types denoted in the function type hints.

### Objective 3: Implement Regression
Complete the functions `make_regression_model` and `model_missing_data` so the missing data will be populated via random forest regreession instead of k-nearest-neighbor imputation. This method should use the sklearn.preprocessing.RandomForestRegressor` class and accept and return the data types denoted in the function type hints.

## Repository Contents

`assignment.py` -- The code entrypoint that should be altered

`data` -- Folder containing the diamond data

`LICENSE` -- The Apache License 2.0

`README.md` -- This file

`requirements.txt` -- Python dependencies

`utils` -- Folder containing the code used to create missing data. Not needed for the challenge but may be helpful