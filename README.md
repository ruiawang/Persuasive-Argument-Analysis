# Persuasive-Argument-Analysis
A toy study into the social dynamics and language characteristics of persuasive arguments as well as openness vs. resistance to persuasion.

This study is heavily inspired by this [paper](https://chenhaot.com/pubs/winning-arguments.pdf).

This project is a toy study into the characteristics and dynamics of persuasive arguments as well as resistance to persuasion based on data from [r/changemyview](https://www.reddit.com/r/changemyview) from 2013-2015.

The dataset used for this study can be found [here](https://chenhaot.com/papers/changemyview.html).

Files:
- *interaction_dynamics.py*: Script to implement and visualize the dynamics of interactions between OP and challenger. Corresponds to Section 3 of the original paper.
- *language holdout calc.py*: Script to calculate the feature values for the holdout data set in the language indicators task.
- *language indicators.py*: Script to implement, calculate, and visualize the dynamics for language characteristics between successful and unsuccessful arguments. Corresponds to Section 4 of the original paper.
- *language predicting.py*: Script that trains and runs predictions on simple Logistic Regression models to predict whether or not a given reply to a post will be successful in changing the OP's mind.
- *persuasion prediction.py*: Script that runs predictions using the GPT-3.5 API on whether not a given OP will have their mind changed based on their post content.
- *resistance to persausion.py*: Script that implements, calculates features, and performs tests to determine characteristics and significance for whether or not a given OP will have their mind changed. Corresponds to Section 5 of the original paper.
- *sample.py*: Script to generate random samples from the original dataset.
- *utils.py*: Several useful utility functions for preprocessing and cleaning of the data.
