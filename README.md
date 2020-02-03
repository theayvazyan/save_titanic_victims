![alt text](https://github.com/theayvazyan/save_titanic_victims/raw/master/img/intro.jpg "Titanic")

# Titanic Passenger Analysis

Based on survival data, predict which passengers will be safe, and which of them need to change the ship using XGBRegressor, Pipelining and Feauture Engineering.

## Feature Engineering

In this data set we have 2 types of data: numerical (Age, Passenger Id, Pclass, SibSp, Parch, Fare) and categorical (Embarked, Sex). 

Let's take a look at missing values:

|  Train set Column name   | Number of null items |
| -----------------------  | -------------------: |
| PassengerId              |    0                 |
| Survived                 |    0                 |
| Pclass                   |    0                 |
| Name                     |    0                 |
| Sex                      |    0                 |
| Age                      |  177                 |
| SibSp                    |    0                 |
| Parch                    |    0                 |
| Ticket                   |    0                 |
| Fare                     |    0                 |
| Cabin                    |  687                 |
| Embarked                 |    2                 |


| Test set Column name     | Number of null items |
| -----------------------  | -------------------: |
| PassengerId              |    0                 |
| Survived                 |    0                 |
| Pclass                   |    0                 |
| Name                     |    0                 |
| Sex                      |    0                 |
| Age                      |   86                 |
| SibSp                    |    0                 |
| Parch                    |    0                 |
| Ticket                   |    0                 |
| Fare                     |    1                 |
| Cabin                    |  327                 |
| Embarked                 |    0                 |

There are some missing values in both of the groups, so we should implement data imputing techniques: mean imputing for numerical values and constant character imputing for categorical. Afterwards, label encoding process has been initiated to convert the categorical group into numerical.

At this point, we have prepared the data for further analysis.

## Understanding the correlations

### Survival rates by sex

![alt text](https://github.com/theayvazyan/save_titanic_victims/raw/master/img/sex_analysis.png "Passenger sex and survival rates")

As it can be seen on the chart, number of male passengers on board were much grater than femail and child passenger, though female survival rate (mean in this case) is much higher. 

![alt text](https://github.com/theayvazyan/save_titanic_victims/raw/master/img/family_analysis.png "With or without family passenger survival rates")

People with families had higher survival rate than people travelling alone. This should also need to be considered while predicting. 

![alt text](https://github.com/theayvazyan/save_titanic_victims/raw/master/img/pclass_analysis.png "Survival rates by ticket class")

Cheaper the class, lower the possibility to survive! When looking at class distribution on the ship scheme, it's clear, that people in first class had more lifeboats, were located in higher floors, hens had more time to escape.

![alt text](https://github.com/theayvazyan/save_titanic_victims/raw/master/img/class_distribution.jpg "Class distribution on board")

To better understand feature correlations, let's take a look at pairwise heatmap and plots:

![alt text](https://github.com/theayvazyan/save_titanic_victims/raw/master/img/pairwise_heatmap.png "Pairwise heatmap")

![alt text](https://github.com/theayvazyan/save_titanic_victims/raw/master/img/pairwise_graph_plot.png "Pairwise graph plots")

## Method description

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. 

### Cons

+ Parallelization of tree construction using all of your CPU cores during training.
+ Distributed Computing for training very large models using a cluster of machines.
+ Out-of-Core Computing for very large datasets that don’t fit into memory.
+ Cache Optimization of data structures and algorithm to make best use of hardware.

thus, resulting in

+ High Execution Speed
+ High Model Performance
