# ABC Rule Miner

A Supervised Explainable Machine Learning Algorithm for Generating High-Precision Decision Rules

## Overview

ABC Rule Miner is a supervised rule-based learning algorithm designed to generate interpretable IF–THEN rules for predicting a target behavioral label.
This algorithm was introduced in the paper : https://www.sciencedirect.com/science/article/pii/S1084804520302368
Unlike Apriori (unsupervised), ABC Rule Miner uses a target column (e.g., Recommendation_Label) to grow branches in a tree-like structure and extract behavior-focused rules.

Major contributions of the paper :

    1. Identify redundancy in associations while producing user behavioral rules based on multi-dimensional contexts.
    2. A rule based ML method (Association Generation Tree) to discover non redundant behavioral associations.
    3. Conducting range of experiments on real smart phone dataset, comparing results of ABC rule miner approach with traditional methods ( Apriori).

## Objectives
    1. Understand the working of ABC Rule Miner.
    2. Implement ABC Rule Miner and Apriori on an E-learning dataset and understand and compare the results.

This project implements:
    
    1. ABC rule generation
    2. Rule pruning
    3. Confidence filtering
    4. Coverage evaluation
    5. Benchmarking against Apriori
    6. Visualizations (rule count vs. confidence threshold)

## Installation

```bash
 git clone https://github.com/Joyee2004/ABC-Rule-Miner.git
 cd ABC-Rule-Miner
```
Now in the experiments directory, run each file (.ipynb)/ code snippet.


## Dataset
Contextual E-Learning Learner Interaction Dataset.

It contains data from e-learning platforms, covering usage details.

Dataset link : https://www.kaggle.com/datasets/zoya77/contextual-e-learning-learner-interaction-dataset


<img width="1590" height="253" alt="Image" src="https://github.com/user-attachments/assets/16a41628-6d03-4301-909a-d640b57dcfd2" />

<img width="1589" height="291" alt="Image" src="https://github.com/user-attachments/assets/7a39c426-6456-4d8b-a520-2640696729ad" />

<img width="1119" height="264" alt="Image" src="https://github.com/user-attachments/assets/2d3fe50f-09f4-4de0-be8e-176724e2778b" />

## ABC Rule Miner working

<img width="1049" height="557" alt="Image" src="https://github.com/user-attachments/assets/8c404155-96ba-409e-ae10-0f7de540e37e" />

    1. Input:

        i. Training dataset

        ii. Target behavior column

        iii. Confidence threshold t

    2. Recursive Branch Growth: 
        i. At each step, select an attribute that best separates data toward the target label.

        ii. Expand branches similar to a decision tree.

    3. Rule Extraction: Convert each path from root → leaf into an IF–THEN rule.

    4. Pruning:

        i. Remove rules below confidence threshold t.

        ii. Remove redundant rule subsets.

    5. Evaluation:

        i. Apply rules on test data.

        ii. Compute Accuracy, Precision, Recall, F1.

## Results

####  Results for Apriori algorithm:

<img width="1349" height="580" alt="Image" src="https://github.com/user-attachments/assets/a1568a2e-a672-4601-8e01-376509141464" />


Observations:

    From confidence 0.1 to 0.5 , there is no change in result
    From 0.6 to 0.8 , performance increases.
    From 0.8 to 0.9, performance decreases.

#### Results for ABC Rule Miner :

t | Accuracy | Precision | Recall | F1
-- | -- | -- | -- | --
0.1 | 0.631 | 0.631 | 1 | 0.774
0.2 | 0.631 | 0.631 | 1 | 0.774
0.3 | 0.631 | 0.631 | 1 | 0.774
0.4 | 0.631 | 0.631 | 1 | 0.774
0.5 | 0.631 | 0.631 | 1 | 0.774
0.6 | 0.943 | 0.917 | 1 | 0.957
0.7 | 0.943 | 0.917 | 1 | 0.957
0.8 | 0.943 | 0.917 | 1 | 0.957
0.9 | 0.887 | 0.93 | 0.888 | 0.909

<img width="1546" height="922" alt="Image" src="https://github.com/user-attachments/assets/97e26405-641d-4f2a-9d1d-046c1ae974a3" />

### Number of rules vs Confidence threshold

<img width="1226" height="834" alt="Image" src="https://github.com/user-attachments/assets/420d7564-9785-440f-8aba-097e78c8606b" />


Observations:

    1. Number of rules generated:
        Apriori : increases with decrease in confidence.
        ABC Rule Miner : increases with increase in confidence.
        From 0.1 to 0.5, performance of both are similar, but ABC generates lesser  rules.
    2. Apriori generates redundant rules.


## Conclusion

    1. From confidence threshold 0.1 to 0.5, ABC Rule miner gives same result as that of Apriori but with lesser number of rules.
    2. For ABC rule Miner, an optimal value of t is 0.6 where it outperforms apriori.
    ABC Rule Miner eliminates redundant rules.
    3. A tradeoff between number of rules and performance exists.
