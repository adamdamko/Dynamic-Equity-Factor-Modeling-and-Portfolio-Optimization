# Dynamic Equity Factor Modeling and Portfolio Optimization

## Description
This application is designed to extend standard linear factor modeling in equity investing by engineering dynamic features
across time, and utilizing gradient boosting methods to capture non-linear and feature interaction dynamics.

Everything is performed in Python. 
The primary packages you will need are Pandas, Numpy, Scikit-Learn, and LightGBM.
The model is managed using the Kedro package.
A full list of requirements can be found in under the `requirement.txt` file.

In addition to the Python packages needed, you will also need data.
The basic data requirements are historical pricing data for as many equity securities to which you have access.
The model is modular and is designed to be able to add as many features as you wish (such as price-to-book).
The data used currently come from [SimFin](https://simfin.com/).

The application will take basic data inputs and perform the following:

1. Perform some exploratory data analysis on the data provided.
2. Create dynamic features as inputs to the model.
3. Perform hyperparameter tuning (a set of hyperparameters are already supplied however).
4. Out-of-sample testing (code will have to be modified to fit the data you provide).
5. Portfolio optimization.
6. Portfolio backtesting.
7. Output of selected securities and weights for a long-only portfolio.

This application is robust enough to create a real world portfolio, 
but not overly complex and does not require a large amount of expensive data.
The author hopes others can take this project and build on it with different modeling techniques,
different portfolio optimization techniques, new data, and new feature engineering.
Hopefully, the end result is a production ready machine learning and portfolio optimization application
that can compete with and outperform professionally managed money.


## How to Use
First you can fork the project.

Once you have forked the project and have the package requirements and data input, you can follow the instructions for running a Kedro project found under the `investing` directory.
You can modify the Kedro pipeline to only run certain parts of the project.
You can also add, delete, and/or modify any existing pipelines.



