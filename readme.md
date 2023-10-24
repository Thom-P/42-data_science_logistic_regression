# 42 Data Science: Logistic Regression (DSLR)

The goal of this project is to implement a multi-class linear classifier "from scratch" using logistic regression. From scratch here means that we implement our own data statistics tools, feature scaling, cost function, and gradient descient descent algorithm.

## The story

The Sorting Hat of the infamous Hogwarts school of wizards is not longer working and cannot fulfill his role of sorting the students into the four houses: _Ravenclaw_, _Gryffindor_, _Slytherin_, and _Hufflepuff_. We are provided a dataset containing the entry test scores of 1600 students from the previous years, and the houses they were assigned to. With our muggle datascience and machine learning tools, we need to simulate a Sorting Hat to be able to sort the 400 new students into their best fitting houses based on their entry test scores.

## Tools

We use Python 3.10 and the following libraries: numpy for efficient array/matrix operations, pandas for the manipulation of the dataset, and matplotlib for the visualization.

## Task 1: Data Analysis

We create a program for computing basic statistics of our training dataset. For each of the 13 course/topics, we compute the number of data points (count), mean score, standard deviation, minimum, maximum, median, first and third quartile. Note that this program is basically a reimplementation of the pandas builtin describe() function (that we are not allowed to use).

Usage: 
```sh
python ft_describe.py ../datasets/dataset_train.csv
```


