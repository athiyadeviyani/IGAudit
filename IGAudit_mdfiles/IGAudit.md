# IG Audit
Objective: Using Simple Statistical Tools and Machine Learning to Audit Instagram Accounts for Authenticity

Motivation: During lockdown, businesses have started increasing the use of social media influencers to market their products while their physical outlets are temporary closed. However, it is sad that there are some that will try and game the system for their own good. But in a world where a single influencer's post is worth as much as an average 9-5 Joe's annual salary, influencer marketing fake followers and fake engagement is a price that brands shouldn't have to pay for.

*Inspired by igaudit.io that was taken down by Facebook only recently.*


```python
# Imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from instagram_private_api import Client, ClientCompatPatch
import getpass

import random
```

## Part 1: Understanding and Splitting the Data
Dataset source: https://www.kaggle.com/eswarchandt/is-your-insta-fake-or-genuine

Import the data


```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

Inspect the training data


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.27</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>1000</td>
      <td>955</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>286</td>
      <td>2740</td>
      <td>533</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.10</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
      <td>159</td>
      <td>98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
      <td>679</td>
      <td>414</td>
      <td>651</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>151</td>
      <td>126</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The features in the training data are the following:
- profile pic: does the user have a profile picture?
- nums/length username: ratio of numerical to alphabetical characters in the username
- fullname words: how many words are in the user's full name?
- nums/length fullname: ratio of numerical to alphabetical characters in the full name
- name==username: is the user's full name the same as the username?
- description length: how many characters is in the user's Instagram bio?
- external URL: does the user have an external URL linked to their profile?
- private: is the user private?
- #posts: number of posts
- #followers: number of people following the user
- #follows: number of people the user follows
- fake: if the user is fake, fake=1, else fake=0


```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>576.000000</td>
      <td>5.760000e+02</td>
      <td>576.000000</td>
      <td>576.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.701389</td>
      <td>0.163837</td>
      <td>1.460069</td>
      <td>0.036094</td>
      <td>0.034722</td>
      <td>22.623264</td>
      <td>0.116319</td>
      <td>0.381944</td>
      <td>107.489583</td>
      <td>8.530724e+04</td>
      <td>508.381944</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.458047</td>
      <td>0.214096</td>
      <td>1.052601</td>
      <td>0.125121</td>
      <td>0.183234</td>
      <td>37.702987</td>
      <td>0.320886</td>
      <td>0.486285</td>
      <td>402.034431</td>
      <td>9.101485e+05</td>
      <td>917.981239</td>
      <td>0.500435</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.900000e+01</td>
      <td>57.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>1.505000e+02</td>
      <td>229.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.310000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>81.500000</td>
      <td>7.160000e+02</td>
      <td>589.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>0.920000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>150.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7389.000000</td>
      <td>1.533854e+07</td>
      <td>7500.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 576 entries, 0 to 575
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   profile pic           576 non-null    int64  
     1   nums/length username  576 non-null    float64
     2   fullname words        576 non-null    int64  
     3   nums/length fullname  576 non-null    float64
     4   name==username        576 non-null    int64  
     5   description length    576 non-null    int64  
     6   external URL          576 non-null    int64  
     7   private               576 non-null    int64  
     8   #posts                576 non-null    int64  
     9   #followers            576 non-null    int64  
     10  #follows              576 non-null    int64  
     11  fake                  576 non-null    int64  
    dtypes: float64(2), int64(10)
    memory usage: 54.1 KB



```python
train.shape
```




    (576, 12)



Inspect the test data


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.33</td>
      <td>1</td>
      <td>0.33</td>
      <td>1</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>35</td>
      <td>488</td>
      <td>604</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.00</td>
      <td>5</td>
      <td>0.00</td>
      <td>0</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>35</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.00</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>1</td>
      <td>319</td>
      <td>328</td>
      <td>668</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>143</td>
      <td>0</td>
      <td>1</td>
      <td>273</td>
      <td>14890</td>
      <td>7369</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.50</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>76</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>225</td>
      <td>356</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>120.000000</td>
      <td>1.200000e+02</td>
      <td>120.000000</td>
      <td>120.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.758333</td>
      <td>0.179917</td>
      <td>1.550000</td>
      <td>0.071333</td>
      <td>0.041667</td>
      <td>27.200000</td>
      <td>0.100000</td>
      <td>0.308333</td>
      <td>82.866667</td>
      <td>4.959472e+04</td>
      <td>779.266667</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.429888</td>
      <td>0.241492</td>
      <td>1.187116</td>
      <td>0.209429</td>
      <td>0.200664</td>
      <td>42.588632</td>
      <td>0.301258</td>
      <td>0.463741</td>
      <td>230.468136</td>
      <td>3.816126e+05</td>
      <td>1409.383558</td>
      <td>0.502096</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>6.725000e+01</td>
      <td>119.250000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2.165000e+02</td>
      <td>354.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.330000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.250000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>58.250000</td>
      <td>5.932500e+02</td>
      <td>668.250000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>0.890000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>149.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1879.000000</td>
      <td>4.021842e+06</td>
      <td>7453.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 120 entries, 0 to 119
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   profile pic           120 non-null    int64  
     1   nums/length username  120 non-null    float64
     2   fullname words        120 non-null    int64  
     3   nums/length fullname  120 non-null    float64
     4   name==username        120 non-null    int64  
     5   description length    120 non-null    int64  
     6   external URL          120 non-null    int64  
     7   private               120 non-null    int64  
     8   #posts                120 non-null    int64  
     9   #followers            120 non-null    int64  
     10  #follows              120 non-null    int64  
     11  fake                  120 non-null    int64  
    dtypes: float64(2), int64(10)
    memory usage: 11.4 KB



```python
test.shape
```




    (120, 12)



Check for NULL values


```python
print(train.isna().values.any().sum())
print(test.isna().values.any().sum())
```

    0
    0


Create a correlation matrix for the features in the training data to check for significantly relevant features


```python
fig, ax = plt.subplots(figsize=(15,10))  
corr=train.corr()
sns.heatmap(corr, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10a5f4590>




![png](output_19_1.png)


Split the training set into data and labels


```python
# Labels
train_Y = train.fake
train_Y = pd.DataFrame(train_Y)

# Data
train_X = train.drop(columns='fake')
train_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.27</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>1000</td>
      <td>955</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>286</td>
      <td>2740</td>
      <td>533</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.10</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
      <td>159</td>
      <td>98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
      <td>679</td>
      <td>414</td>
      <td>651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>151</td>
      <td>126</td>
    </tr>
  </tbody>
</table>
</div>



Split the test set into data and labels


```python
# Labels
test_Y = test.fake
test_Y = pd.DataFrame(test_Y)

# Data
test_X = test.drop(columns='fake')
test_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.33</td>
      <td>1</td>
      <td>0.33</td>
      <td>1</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>35</td>
      <td>488</td>
      <td>604</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.00</td>
      <td>5</td>
      <td>0.00</td>
      <td>0</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>35</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.00</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>1</td>
      <td>319</td>
      <td>328</td>
      <td>668</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>143</td>
      <td>0</td>
      <td>1</td>
      <td>273</td>
      <td>14890</td>
      <td>7369</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.50</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>76</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>225</td>
      <td>356</td>
    </tr>
  </tbody>
</table>
</div>



## Part 2: Comparing Classification Models

**Baseline Classifier**
<br>Classify everything as the majority class.


```python
# Baseline classifier
fakes = len([i for i in train.fake if i==1])
auth = len([i for i in train.fake if i==0])
fakes, auth

# classify everything as fake
pred = [1 for i in range(len(test_X))]
pred = np.array(pred)
print("Baseline accuracy: " + str(accuracy_score(pred, test_Y)))
```

    Baseline accuracy: 0.5


**Statistical Method**
<br>Classify all users with a following to follower ratio above a certain threshold as 'fake'.
<br> i.e. a user with 10 follower and 200 followings will be classified as fake if the threshold r=20


```python
# Statistical method
def stat_predict(test_X, r):
    pred = []
    for row in range(len(test_X)):   
        followers = test_X.loc[row]['#followers']
        followings = test_X.loc[row]['#follows']
        if followers == 0:
            followers = 1
        if followings == 0:
            followings == 1

        ratio = followings/followers

        if ratio >= r:
            pred.append(1)
        else:
            pred.append(0)
    
    return np.array(pred)
accuracies = []
for i in [x / 10.0 for x in range(5, 255, 5)]:
    prediction = stat_predict(test_X, i)
    accuracies.append(accuracy_score(prediction, test_Y))

f, ax = plt.subplots(figsize=(20,10))
plt.plot([x / 10.0 for x in range(5, 255, 5)], accuracies)
plt.plot([2.5 for i in range(len(accuracies))], accuracies, color='red')
plt.title("Accuracy for different thresholds", size=30)
plt.xlabel('Ratio', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
print("Maximum Accuracy for the statistical method: " + str(max(accuracies)))
```

    Maximum Accuracy for the statistical method: 0.7



![png](output_28_1.png)


**Logistic Regression**


```python
lm = LogisticRegression()

# Train the model
model1 = lm.fit(train_X, train_Y)

# Make a prediction
lm_predict = model1.predict(test_X)
```

    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)



```python
# Compute the accuracy of the model
acc = accuracy_score(lm_predict, test_Y)
print("Logistic Regression accuracy: " + str(acc))
```

    Logistic Regression accuracy: 0.9083333333333333


**KNN Classifier**


```python
accuracies = []

# Compare the accuracies of using the KNN classifier with different number of neighbors
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    model_2 = knn.fit(train_X,train_Y)
    knn_predict = model_2.predict(test_X)
    accuracy = accuracy_score(knn_predict,test_Y)
    accuracies.append(accuracy)

max_acc = (0, 0)
for i in range(1, 10):
    if accuracies[i-1] > max_acc[1]:
        max_acc = (i, accuracies[i-1])

max_acc

f, ax = plt.subplots(figsize=(20,10))
plt.plot([i for i in range(1,10)], accuracies)
plt.plot([7 for i in range(len(accuracies))], accuracies, color='red')
plt.title("Accuracy for different n-neighbors", size=30)
plt.xlabel('Number of neighbors', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

print("The highest accuracy obtained using KNN is " + str(max_acc[1]) + " achieved by a value of n=" + str(max_acc[0]))
```

    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      
    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      


    The highest accuracy obtained using KNN is 0.8666666666666667 achieved by a value of n=7



![png](output_33_2.png)


**Decision Tree Classifier**


```python
DT = DecisionTreeClassifier()

# Train the model
model3 = DT.fit(train_X, train_Y)

# Make a prediction
DT_predict = model3.predict(test_X)
```


```python
# Compute the accuracy of the model
acc = accuracy_score(DT_predict, test_Y)
print("Decision Tree accuracy: " + str(acc))
```

    Decision Tree accuracy: 0.9


**Random Forest Classifier**


```python
rfc = RandomForestClassifier()

# Train the model
model_4 = rfc.fit(train_X, train_Y)

# Make a prediction
rfc_predict = model_4.predict(test_X)
```

    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      after removing the cwd from sys.path.



```python
# Compute the accuracy of the model
acc = accuracy_score(rfc_predict, test_Y)
print("Random Forest accuracy: " + str(acc))
```

    Random Forest accuracy: 0.925


## Part 3: Obtaining Instagram Data
We are going to use the hassle-free unofficial Instagram API. <br> To install: ```$ pip install git+https://git@github.com/ping/instagram_private_api.git@1.6.0```

Log in to your Instagram account (preferably not your personal one! I created one just for this project 😉)


```python
def login():
    username = input("username: ")
    password = getpass.getpass("password: ")
    api = Client(username, password)
    return api

api = login()
```

    username: ins.tapolice
    password: ········


Get the Instagram user ID


```python
def get_ID(username):
    return api.username_info(username)['user']['pk']
```


```python
# The user used for the experiment below is anonymised!
# i.e. this cell was run and then changed to protect the user's anonymity
userID = get_ID('<USERNAME HERE>') 
```

The API needs some sort of rank to query followers, posts, etc.


```python
rank = api.generate_uuid()
```

Get the user's list follower usernames (this may take a while, depending on how many followers the user have)


```python
def get_followers(userID, rank):
    followers = []
    next_max_id = True
    
    while next_max_id:
        if next_max_id == True: next_max_id=''
        f = api.user_followers(userID, rank, max_id=next_max_id)
        followers.extend(f.get('users', []))
        next_max_id = f.get('next_max_id', '')
    
    user_fer = [dic['username'] for dic in followers]
    
    return user_fer
```


```python
followers = get_followers(userID, rank)
```


```python
# You can check the number of followers if you'd like to
# len(followers)
```

## Part 4: Preparing the Data

Inspect the data (and what other data can you obtain from it) and compare it with the train and test tables above. Find out what you need to do to obtain the features for a data point in order to make a prediction.

Recall that the features for a data point are the following:
- profile pic: does the user have a profile picture?
- nums/length username: ratio of numerical to alphabetical characters in the username
- fullname words: how many words are in the user's full name?
- nums/length fullname: ratio of numerical to alphabetical characters in the full name
- name==username: is the user's full name the same as the username?
- description length: how many characters is in the user's Instagram bio?
- external URL: does the user have an external URL linked to their profile?
- private: is the user private?
- #posts: number of posts
- #followers: number of people following the user
- #follows: number of people the user follows
- fake: if the user is fake, fake=1, else fake=0


```python
# This will print the first follower username on the list
# print(followers[0])
```


```python
# This will get the information on a certain user
info = api.user_info(get_ID(followers[0]))['user']

# Check what information is available for one particular user
info.keys()
```




    dict_keys(['pk', 'username', 'full_name', 'is_private', 'profile_pic_url', 'profile_pic_id', 'is_verified', 'has_anonymous_profile_picture', 'media_count', 'geo_media_count', 'follower_count', 'following_count', 'following_tag_count', 'biography', 'biography_with_entities', 'external_url', 'external_lynx_url', 'total_igtv_videos', 'total_clips_count', 'total_ar_effects', 'usertags_count', 'is_favorite', 'is_favorite_for_stories', 'is_favorite_for_highlights', 'live_subscription_status', 'is_interest_account', 'has_chaining', 'hd_profile_pic_versions', 'hd_profile_pic_url_info', 'mutual_followers_count', 'has_highlight_reels', 'can_be_reported_as_fraud', 'is_eligible_for_smb_support_flow', 'smb_support_partner', 'smb_delivery_partner', 'smb_donation_partner', 'smb_support_delivery_partner', 'displayed_action_button_type', 'direct_messaging', 'fb_page_call_to_action_id', 'address_street', 'business_contact_method', 'category', 'city_id', 'city_name', 'contact_phone_number', 'is_call_to_action_enabled', 'latitude', 'longitude', 'public_email', 'public_phone_country_code', 'public_phone_number', 'zip', 'instagram_location_id', 'is_business', 'account_type', 'professional_conversion_suggested_account_type', 'can_hide_category', 'can_hide_public_contacts', 'should_show_category', 'should_show_public_contacts', 'personal_account_ads_page_name', 'personal_account_ads_page_id', 'include_direct_blacklist_status', 'is_potential_business', 'show_post_insights_entry_point', 'is_bestie', 'has_unseen_besties_media', 'show_account_transparency_details', 'show_leave_feedback', 'robi_feedback_source', 'auto_expand_chaining', 'highlight_reshare_disabled', 'is_memorialized', 'open_external_url_with_in_app_browser'])



You can see that we have pretty much all the features to make a user data point for prediction, but we need to filter and extract them, and perform some very minor calculations. The following function will do just that:


```python
def get_data(info):
    
    """Extract the information from the returned JSON.
    
    This function will return the following array:
        data = [profile pic,
                nums/length username,
                full name words,
                nums/length full name,
                name==username,
                description length,
                external URL,
                private,
                #posts,
                #followers,
                #followings]
    """
    
    data = []
    
    # Does the user have a profile photo?
    profile_pic = not info['has_anonymous_profile_picture']
    if profile_pic == True:
        profile_pic = 1
    else:
        profile_pic = 0
    data.append(profile_pic)
    
    # Ratio of number of numerical chars in username to its length
    username = info['username']
    uname_ratio = len([x for x in username if x.isdigit()]) / float(len(username))
    data.append(uname_ratio)
    
    # Full name in word tokens
    full_name = info['full_name']
    fname_tokens = len(full_name.split(' '))
    data.append(fname_tokens)
    
    # Ratio of number of numerical characters in full name to its length
    if len(full_name) == 0:
        fname_ratio = 0
    else:
        fname_ratio = len([x for x in full_name if x.isdigit()]) / float(len(full_name))
    data.append(fname_ratio)
    
    # Is name == username?
    name_eq_uname = (full_name == username)
    if name_eq_uname == True:
        name_eq_uname = 1
    else:
        name_eq_uname = 0
    data.append(name_eq_uname)
    
    # Number of characters on user bio 
    bio_length = len(info['biography'])
    data.append(bio_length)
    
    # Does the user have an external URL?
    ext_url = info['external_url'] != ''
    if ext_url == True:
        ext_url = 1
    else:
        ext_url = 0
    data.append(ext_url)
    
    # Is the user private or no?
    private = info['is_private']
    if private == True:
        private = 1
    else:
        private = 0
    data.append(private)
    
    # Number of posts
    posts = info['media_count']
    data.append(posts)
    
    # Number of followers
    followers = info['follower_count']
    data.append(followers)
    
    # Number of followings
    followings = info['following_count']
    data.append(followings)
    
  
    return data
```


```python
# Check if the function returns as expected
get_data(info)
```




    [1, 0.0, 3, 0.0, 0, 118, 1, 0, 589, 22227, 510]



Unfortunately the Instagram Private API has a very limited number of API calls per hour so we will not be able to analyse *all* of the user's followers. 

Fortunately, I took Statistics and learned that **random sampling** is useful to cull a smaller sample size from a larger population and use it to research and make generalizations about the larger group. 

This will allow us to make user authenticity approximations despite the API limitations and still have a data that is representative of the user's followers.


```python
# Get a random sample of 50 followers
random_followers = random.sample(followers, 50)
```

Get user information for each follower


```python
f_infos = []

for follower in random_followers:
    info = api.user_info(get_ID(follower))['user']
    f_infos.append(info)
```

Extract the relevant features


```python
f_table = []

for info in f_infos:
    f_table.append(get_data(info))
    
f_table
```




    [[1, 0.0, 3, 0.0, 0, 43, 0, 1, 108, 788, 764],
     [1, 0.0, 1, 0, 0, 45, 0, 0, 1, 252, 483],
     [1, 0.0, 3, 0.0, 0, 90, 0, 0, 536, 1818, 7486],
     [1, 0.5, 3, 0.0, 0, 0, 0, 0, 157, 148, 813],
     [1, 0.0, 1, 0.0, 0, 102, 0, 1, 24, 481, 592],
     [1, 0.0, 1, 0.0, 0, 59, 0, 1, 19, 773, 3639],
     [1, 0.0, 1, 0, 0, 8, 0, 1, 0, 3, 3639],
     [1, 0.0, 3, 0.0, 0, 90, 1, 0, 27, 63, 19],
     [1, 0.0, 4, 0.0, 0, 148, 0, 1, 458, 682, 436],
     [1, 0.0, 2, 0.0, 0, 0, 0, 1, 35, 1054, 1046],
     [1, 0.36363636363636365, 1, 0.0, 0, 96, 0, 1, 96, 50, 98],
     [1, 0.0, 1, 0.0, 0, 0, 0, 1, 2, 10, 202],
     [1, 0.0, 2, 0.0, 0, 135, 1, 1, 159, 52, 240],
     [1, 0.0, 1, 0.0, 0, 20, 0, 0, 87, 1864, 692],
     [1, 0.0, 1, 0.0, 0, 0, 0, 1, 35, 275, 2039],
     [1, 0.0625, 3, 0.0, 0, 98, 0, 0, 9, 98, 847],
     [1, 0.0, 3, 0.0, 0, 92, 0, 1, 10, 11, 46],
     [1, 0.0, 2, 0.0, 0, 69, 0, 1, 16, 2686, 6570],
     [1, 0.0, 2, 0.0, 0, 68, 0, 1, 31, 18, 64],
     [1, 0.0, 3, 0.0, 0, 6, 0, 0, 27, 1628, 1037],
     [1, 0.0, 1, 0, 0, 2, 0, 0, 21, 1730, 1298],
     [0, 0.18181818181818182, 2, 0.0, 0, 0, 0, 1, 219, 183, 275],
     [1, 0.0, 2, 0.0, 0, 38, 0, 0, 11, 645, 4452],
     [1, 0.0, 2, 0.0, 0, 30, 1, 0, 42, 1258, 952],
     [1, 0.0, 1, 0.0, 0, 9, 0, 0, 2, 629, 485],
     [1, 0.23529411764705882, 1, 0.0, 0, 62, 0, 1, 12, 1270, 951],
     [1, 0.0, 1, 0.0, 0, 86, 0, 0, 299, 1669, 1133],
     [1, 0.0, 2, 0.0, 0, 14, 0, 0, 11, 753, 853],
     [1, 0.2, 2, 0.0, 0, 9, 0, 0, 0, 213, 700],
     [1, 0.0, 1, 0.0, 0, 133, 0, 1, 11, 28, 169],
     [1, 0.0, 2, 0.0, 0, 0, 0, 1, 3, 1395, 794],
     [1, 0.0, 2, 0.0, 0, 0, 0, 0, 71, 831, 1024],
     [1, 0.0, 3, 0.0, 0, 29, 0, 0, 61, 680, 566],
     [1, 0.0, 2, 0.0, 0, 64, 0, 0, 1729, 6114, 5758],
     [1, 0.0, 2, 0.0, 0, 17, 0, 0, 73, 2104, 7091],
     [1, 0.0, 3, 0.0, 0, 36, 0, 1, 20, 728, 4139],
     [1, 0.0, 2, 0.0, 0, 106, 0, 1, 23, 83, 458],
     [1, 0.0, 2, 0.0, 0, 31, 0, 1, 78, 2035, 1035],
     [1, 0.0, 2, 0.0, 0, 35, 0, 1, 12, 11549, 712],
     [1, 0.0, 3, 0.08333333333333333, 0, 100, 0, 1, 56, 39, 190],
     [1, 0.13333333333333333, 1, 0.0, 0, 103, 0, 1, 109, 1053, 6221],
     [1, 0.0, 1, 0.0, 0, 0, 0, 0, 49, 412, 520],
     [1, 0.0, 1, 0, 0, 7, 0, 0, 110, 317, 334],
     [1, 0.0, 1, 0.0, 0, 31, 1, 0, 141, 2490, 1043],
     [1, 0.18181818181818182, 2, 0.0, 0, 35, 1, 0, 320, 2345, 861],
     [1, 0.0, 3, 0.0, 0, 115, 0, 1, 1336, 1018, 1208],
     [1, 0.0, 1, 0.0, 0, 0, 0, 1, 39, 37, 611],
     [1, 0.0, 1, 0.0, 0, 0, 0, 1, 0, 513, 633],
     [1, 0.0, 2, 0.0, 0, 46, 0, 0, 23, 83, 306],
     [1, 0.0, 1, 0.0, 0, 0, 0, 0, 30, 126, 372]]



Create a pandas dataframe


```python
test_data = pd.DataFrame(f_table,
                         columns = ['profile pic', 
                                    'nums/length username', 
                                    'fullname words',
                                    'nums/length fullname',
                                    'name==username',
                                    'description length',
                                    'external URL',
                                    'private',
                                    '#posts',
                                    '#followers',
                                    '#follows'])
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>43</td>
      <td>0</td>
      <td>1</td>
      <td>108</td>
      <td>788</td>
      <td>764</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>252</td>
      <td>483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>536</td>
      <td>1818</td>
      <td>7486</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.500000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>157</td>
      <td>148</td>
      <td>813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>102</td>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>481</td>
      <td>592</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>59</td>
      <td>0</td>
      <td>1</td>
      <td>19</td>
      <td>773</td>
      <td>3639</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>3639</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>90</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>63</td>
      <td>19</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
      <td>148</td>
      <td>0</td>
      <td>1</td>
      <td>458</td>
      <td>682</td>
      <td>436</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>35</td>
      <td>1054</td>
      <td>1046</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.363636</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>1</td>
      <td>96</td>
      <td>50</td>
      <td>98</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>202</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>135</td>
      <td>1</td>
      <td>1</td>
      <td>159</td>
      <td>52</td>
      <td>240</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>87</td>
      <td>1864</td>
      <td>692</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>35</td>
      <td>275</td>
      <td>2039</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0.062500</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>98</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>98</td>
      <td>847</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>92</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>11</td>
      <td>46</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
      <td>1</td>
      <td>16</td>
      <td>2686</td>
      <td>6570</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>68</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>18</td>
      <td>64</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>1628</td>
      <td>1037</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>1730</td>
      <td>1298</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0.181818</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>219</td>
      <td>183</td>
      <td>275</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>645</td>
      <td>4452</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>0</td>
      <td>42</td>
      <td>1258</td>
      <td>952</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>629</td>
      <td>485</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0.235294</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>62</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1270</td>
      <td>951</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>86</td>
      <td>0</td>
      <td>0</td>
      <td>299</td>
      <td>1669</td>
      <td>1133</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>753</td>
      <td>853</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0.200000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>213</td>
      <td>700</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>133</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>28</td>
      <td>169</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1395</td>
      <td>794</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>71</td>
      <td>831</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>61</td>
      <td>680</td>
      <td>566</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>1729</td>
      <td>6114</td>
      <td>5758</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>73</td>
      <td>2104</td>
      <td>7091</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>20</td>
      <td>728</td>
      <td>4139</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>106</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>83</td>
      <td>458</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>78</td>
      <td>2035</td>
      <td>1035</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>11549</td>
      <td>712</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.083333</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>1</td>
      <td>56</td>
      <td>39</td>
      <td>190</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>0.133333</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>103</td>
      <td>0</td>
      <td>1</td>
      <td>109</td>
      <td>1053</td>
      <td>6221</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>412</td>
      <td>520</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>110</td>
      <td>317</td>
      <td>334</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>31</td>
      <td>1</td>
      <td>0</td>
      <td>141</td>
      <td>2490</td>
      <td>1043</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>0.181818</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>320</td>
      <td>2345</td>
      <td>861</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>115</td>
      <td>0</td>
      <td>1</td>
      <td>1336</td>
      <td>1018</td>
      <td>1208</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>39</td>
      <td>37</td>
      <td>611</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>513</td>
      <td>633</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>83</td>
      <td>306</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>126</td>
      <td>372</td>
    </tr>
  </tbody>
</table>
</div>



## Part 5: Make the prediction!
In part 2, we have compared the different classifiers and found that the Random Forest Classifier had the highest accuracy at 92.5%. Therefore, we are going to use this classifier to make the prediction.


```python
rfc = RandomForestClassifier()

# Train the model
# We've done this in Part 2 but I'm redoing it here for coherence ☺️
rfc_model = rfc.fit(train_X, train_Y)
```

    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      """



```python
rfc_labels = rfc_model.predict(test_data)
rfc_labels
```




    array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0])



Calculate the number of fake accounts in the random sample of 50 followers


```python
no_fakes = len([x for x in rfc_labels if x==1])
```

Calculate the Instagram user's authenticity,
    <br>where authenticity = (#followers - #fakes)*100 / #followers


```python
authenticity = (len(random_followers) - no_fakes) * 100 / len(random_followers)
print("User X's Instagram Followers is " + str(authenticity) + "% authentic.")
```

    User X's Instagram Followers is 82.0% authentic.


## Part 6: Extension - Fake Likes
The method above can also be extended to check fake likes within a post.

Get the user's posts


```python
def get_user_posts(userID, min_posts_to_be_retrieved):
    # Retrieve all posts from my profile
    my_posts = []
    has_more_posts = True
    max_id = ''
    
    while has_more_posts:
        feed = api.user_feed(userID, max_id=max_id)
        if feed.get('more_available') is not True:
            has_more_posts = False 
            
        max_id = feed.get('next_max_id', '')
        my_posts.extend(feed.get('items'))
        
        # time.sleep(2) to avoid flooding
        
        if len(my_posts) > min_posts_to_be_retrieved:
            print('Total posts retrieved: ' + str(len(my_posts)))
            return my_posts
            
        if has_more_posts:
            print(str(len(my_posts)) + ' posts retrieved so far...')
           
    print('Total posts retrieved: ' + str(len(my_posts)))
    
    return my_posts
```


```python
posts = get_user_posts(userID, 10)
```

    Total posts retrieved: 18


Pick one post to analyse (here I'm just going to pick by random)


```python
random_post = random.sample(posts, 1)
```

Get post likers


```python
random_post[0].keys()
```




    dict_keys(['taken_at', 'pk', 'id', 'device_timestamp', 'media_type', 'code', 'client_cache_key', 'filter_type', 'carousel_media_count', 'carousel_media', 'can_see_insights_as_brand', 'location', 'lat', 'lng', 'user', 'can_viewer_reshare', 'caption_is_edited', 'comment_likes_enabled', 'comment_threading_enabled', 'has_more_comments', 'next_max_id', 'max_num_visible_preview_comments', 'preview_comments', 'can_view_more_preview_comments', 'comment_count', 'inline_composer_display_condition', 'inline_composer_imp_trigger_time', 'like_count', 'has_liked', 'top_likers', 'photo_of_you', 'usertags', 'caption', 'can_viewer_save', 'organic_tracking_token'])




```python
likers = api.media_likers(random_post[0]['id'])
```

Get a list of usernames


```python
likers_usernames = [liker['username'] for liker in likers['users']]
```

Get a random sample of 50 users


```python
random_likers = random.sample(likers_usernames, 50)
```

Retrieve the information for the 50 users


```python
l_infos = []

for liker in random_likers:
    info = api.user_info(get_ID(liker))['user']
    l_infos.append(info)
```


```python
l_table = []

for info in l_infos:
    l_table.append(get_data(info))

l_table
```




    [[1, 0.0, 1, 0, 0, 30, 0, 0, 6, 21, 177],
     [1, 0.0, 1, 0.0, 0, 69, 0, 1, 131, 942, 1229],
     [1, 0.0, 2, 0.0, 0, 83, 0, 1, 609, 1558, 2925],
     [1, 0.0, 1, 0.0, 0, 39, 0, 0, 851, 2940, 1255],
     [1, 0.0, 1, 0.0, 0, 36, 1, 0, 106, 1626, 1050],
     [0, 0.0, 1, 0, 0, 0, 0, 1, 7, 371, 350],
     [1, 0.0, 2, 0.0, 0, 96, 1, 0, 405, 1656, 2843],
     [1, 0.0, 2, 0.0, 0, 5, 1, 0, 9, 1363, 854],
     [1, 0.0, 1, 0, 0, 1, 0, 1, 5, 433, 371],
     [1, 0.0, 6, 0.0, 0, 93, 1, 0, 73, 1356, 1081],
     [1, 0.0, 3, 0.0, 0, 80, 1, 1, 188, 966, 966],
     [1, 0.0, 3, 0.0, 0, 0, 0, 1, 156, 1401, 1249],
     [1, 0.0, 2, 0.0, 0, 118, 1, 0, 115, 6557, 2423],
     [1, 0.0, 1, 0.0, 0, 12, 0, 0, 84, 1552, 661],
     [1, 0.0, 1, 0.0, 0, 80, 0, 0, 99, 1413, 2479],
     [1, 0.0, 1, 0.0, 0, 23, 0, 1, 12, 1116, 1031],
     [1, 0.0, 1, 0.0, 0, 20, 0, 0, 87, 1864, 692],
     [1, 0.0, 3, 0.0, 0, 62, 1, 0, 17, 1266, 1107],
     [1, 0.0, 2, 0.0, 0, 20, 0, 1, 15, 636, 579],
     [1, 0.0, 4, 0.0, 0, 17, 0, 1, 127, 546, 536],
     [1, 0.0, 1, 0.0, 0, 18, 0, 0, 5, 918, 678],
     [1, 0.2857142857142857, 1, 0.0, 0, 0, 0, 1, 0, 20, 35],
     [1, 0.0, 2, 0.0, 0, 8, 0, 0, 39, 1490, 1321],
     [1, 0.0, 2, 0.0, 0, 0, 0, 0, 10, 519, 547],
     [1, 0.0, 2, 0.0, 0, 0, 0, 0, 43, 933, 1101],
     [1, 0.0, 2, 0.0, 0, 10, 0, 1, 19, 613, 612],
     [1, 0.25, 3, 0.0, 0, 139, 1, 0, 104, 1738, 999],
     [1, 0.0, 3, 0.0, 0, 42, 1, 0, 17, 2973, 1339],
     [1, 0.0, 1, 0.0, 0, 20, 0, 1, 107, 749, 857],
     [1, 0.0, 4, 0.0, 0, 119, 1, 0, 655, 675, 1904],
     [1, 0.0, 1, 0.0, 0, 103, 1, 0, 48, 10075, 2379],
     [1, 0.0, 1, 0.0, 0, 0, 0, 0, 12, 534, 563],
     [1, 0.0, 1, 0, 0, 0, 0, 1, 58, 2220, 1418],
     [1, 0.0, 1, 0.0, 0, 11, 1, 1, 18, 775, 514],
     [1, 0.0, 3, 0.0, 0, 30, 0, 0, 10, 1070, 1364],
     [1, 0.0, 1, 0.0, 0, 18, 0, 0, 108, 1148, 832],
     [1, 0.0, 2, 0.0, 0, 133, 0, 1, 52, 394, 432],
     [1, 0.0, 1, 0, 0, 30, 1, 0, 48, 3441, 1293],
     [1, 0.0, 2, 0.0, 0, 40, 1, 0, 1434, 1642, 1684],
     [1, 0.0, 1, 0.0, 0, 64, 1, 0, 33, 17955, 781],
     [1, 0.0, 2, 0.0, 0, 91, 1, 1, 217, 1014, 1409],
     [1, 0.0, 1, 0, 0, 0, 0, 1, 1, 1347, 872],
     [1, 0.3076923076923077, 1, 0.0, 0, 0, 0, 0, 59, 161, 544],
     [1, 0.0, 3, 0.0, 0, 141, 1, 1, 274, 922, 913],
     [1, 0.0, 1, 0.0, 0, 69, 1, 0, 69, 904, 596],
     [1, 0.0, 1, 0.0, 0, 42, 0, 0, 598, 1877, 6379],
     [1, 0.0, 2, 0.0, 0, 4, 0, 1, 11, 660, 643],
     [1, 0.0, 2, 0.0, 0, 24, 0, 0, 6, 345, 358],
     [1, 0.0, 2, 0.0, 0, 29, 0, 0, 23, 293, 538],
     [1, 0.0, 1, 0.0, 0, 10, 1, 1, 3, 690, 549]]




```python
# Generate pandas dataframe 
l_test_data = pd.DataFrame(l_table,
                         columns = ['profile pic', 
                                    'nums/length username', 
                                    'fullname words',
                                    'nums/length fullname',
                                    'name==username',
                                    'description length',
                                    'external URL',
                                    'private',
                                    '#posts',
                                    '#followers',
                                    '#follows'])
l_test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>21</td>
      <td>177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
      <td>1</td>
      <td>131</td>
      <td>942</td>
      <td>1229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>83</td>
      <td>0</td>
      <td>1</td>
      <td>609</td>
      <td>1558</td>
      <td>2925</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>851</td>
      <td>2940</td>
      <td>1255</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>36</td>
      <td>1</td>
      <td>0</td>
      <td>106</td>
      <td>1626</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>371</td>
      <td>350</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>96</td>
      <td>1</td>
      <td>0</td>
      <td>405</td>
      <td>1656</td>
      <td>2843</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>1363</td>
      <td>854</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>433</td>
      <td>371</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0.0</td>
      <td>0</td>
      <td>93</td>
      <td>1</td>
      <td>0</td>
      <td>73</td>
      <td>1356</td>
      <td>1081</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>188</td>
      <td>966</td>
      <td>966</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>156</td>
      <td>1401</td>
      <td>1249</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>118</td>
      <td>1</td>
      <td>0</td>
      <td>115</td>
      <td>6557</td>
      <td>2423</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>84</td>
      <td>1552</td>
      <td>661</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>1413</td>
      <td>2479</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1116</td>
      <td>1031</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>87</td>
      <td>1864</td>
      <td>692</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>62</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>1266</td>
      <td>1107</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>15</td>
      <td>636</td>
      <td>579</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>1</td>
      <td>127</td>
      <td>546</td>
      <td>536</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>918</td>
      <td>678</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>0.285714</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>35</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>1490</td>
      <td>1321</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>519</td>
      <td>547</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43</td>
      <td>933</td>
      <td>1101</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>19</td>
      <td>613</td>
      <td>612</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0.250000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>139</td>
      <td>1</td>
      <td>0</td>
      <td>104</td>
      <td>1738</td>
      <td>999</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>42</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>2973</td>
      <td>1339</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>107</td>
      <td>749</td>
      <td>857</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>119</td>
      <td>1</td>
      <td>0</td>
      <td>655</td>
      <td>675</td>
      <td>1904</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>103</td>
      <td>1</td>
      <td>0</td>
      <td>48</td>
      <td>10075</td>
      <td>2379</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>534</td>
      <td>563</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>58</td>
      <td>2220</td>
      <td>1418</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>775</td>
      <td>514</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1070</td>
      <td>1364</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>108</td>
      <td>1148</td>
      <td>832</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>133</td>
      <td>0</td>
      <td>1</td>
      <td>52</td>
      <td>394</td>
      <td>432</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>0</td>
      <td>48</td>
      <td>3441</td>
      <td>1293</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>1434</td>
      <td>1642</td>
      <td>1684</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>64</td>
      <td>1</td>
      <td>0</td>
      <td>33</td>
      <td>17955</td>
      <td>781</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>91</td>
      <td>1</td>
      <td>1</td>
      <td>217</td>
      <td>1014</td>
      <td>1409</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1347</td>
      <td>872</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>0.307692</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>161</td>
      <td>544</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>141</td>
      <td>1</td>
      <td>1</td>
      <td>274</td>
      <td>922</td>
      <td>913</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>69</td>
      <td>1</td>
      <td>0</td>
      <td>69</td>
      <td>904</td>
      <td>596</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>598</td>
      <td>1877</td>
      <td>6379</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>660</td>
      <td>643</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>345</td>
      <td>358</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>293</td>
      <td>538</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>690</td>
      <td>549</td>
    </tr>
  </tbody>
</table>
</div>



Finally, make the prediction!


```python
rfc = RandomForestClassifier()
rfc_model = rfc.fit(train_X, train_Y)
rfc_labels_likes = rfc_model.predict(l_test_data)
rfc_labels_likes
```

    /Users/athiyadeviyani/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      





    array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0])



Calculate the fake accounts that liked the user's media


```python
no_fake_likes = len([x for x in rfc_labels_likes if x==1])
```

Calculate the media likes authenticity


```python
media_authenticity = (len(random_likers) - no_fake_likes) * 100 / len(random_likers)
print("The media with the ID:XXXXX has " + str(media_authenticity) + "% authentic likes.")
```

    The media with the ID:XXXXX has 92.0% authentic likes.


## Part 7: Comparison With Another User
I have specifically chosen user X because I trusted their social media 'game' and seemed to have a loyal and engaged following. Let's compare their metrics with a user Y, a user that has a noticable follower growth spike when examined on SocialBlade.

I am going to skip the explanation here because it's just a repetition of the steps performed on user X.


```python
# Re-login because of API call limits 
api = login()
```

    username: ins.tafakebusters
    password: ········



```python
userID_y = get_ID('<USERNAME>')
```


```python
rank = api.generate_uuid()
```

**USER Y FOLLOWERS ANALYSIS**


```python
y_followers = get_followers(userID_y, rank)
```


```python
y_random_followers = random.sample(y_followers, 50)
```


```python
y_infos = []

for follower in y_random_followers:
    info = api.user_info(get_ID(follower))['user']
    y_infos.append(info)
```


```python
y_table = []

for info in y_infos:
    y_table.append(get_data(info))
    
y_table
```




    [[1, 0.14285714285714285, 1, 0.0, 0, 0, 0, 0, 16, 32, 1549],
     [1, 0.2222222222222222, 1, 0.0, 0, 0, 0, 1, 15, 337, 2058],
     [1, 0.25, 2, 0.0, 0, 0, 0, 0, 5, 310, 6343],
     [1, 0.0, 4, 0.0, 0, 97, 0, 0, 1, 14107, 7514],
     [1, 0.36363636363636365, 2, 0.0, 0, 0, 0, 0, 16, 8, 1050],
     [1, 0.25, 2, 0.0, 0, 13, 0, 0, 15, 87, 6741],
     [1, 0.0, 1, 0, 0, 0, 0, 1, 21, 24, 5862],
     [1, 0.0, 1, 0, 0, 13, 0, 1, 27, 1289, 689],
     [1, 0.0, 1, 0.0, 0, 29, 0, 1, 0, 31, 148],
     [1, 0.0, 1, 0, 0, 119, 0, 0, 32, 636, 1293],
     [1, 0.0, 4, 0.0, 0, 20, 0, 0, 144, 3617, 1346],
     [1, 0.21428571428571427, 2, 0.0, 0, 0, 0, 0, 17, 71, 7495],
     [1, 0.13333333333333333, 2, 0.0, 0, 113, 0, 1, 3, 305, 303],
     [0, 0.4444444444444444, 2, 0.0, 0, 0, 0, 1, 1, 63, 283],
     [1, 0.0, 3, 0.0, 0, 0, 0, 0, 17, 115, 7506],
     [0, 0.0625, 2, 0.0, 0, 0, 0, 1, 272, 1446, 2362],
     [1, 0.15384615384615385, 2, 0.0, 0, 0, 0, 0, 6, 1150, 732],
     [1, 0.0, 2, 0.0, 0, 0, 0, 0, 15, 60, 1631],
     [1, 0.0, 1, 0, 0, 13, 0, 0, 15, 11, 221],
     [1, 0.0, 1, 0, 0, 1, 0, 1, 0, 21, 23],
     [1, 0.23076923076923078, 1, 0, 0, 0, 0, 0, 0, 4, 173],
     [1, 0.25, 1, 0.0, 0, 20, 0, 0, 1, 29, 457],
     [1, 0.5, 1, 0.0, 0, 0, 0, 0, 1, 831, 5424],
     [1, 0.0, 3, 0.0, 0, 150, 1, 0, 158, 7063, 1355],
     [1, 0.0, 1, 0.0, 0, 0, 0, 1, 15, 39, 2045],
     [1, 0.0, 4, 0.05555555555555555, 0, 127, 0, 0, 196, 486, 198],
     [1, 0.0, 1, 0.0, 0, 76, 0, 1, 7, 509, 372],
     [1, 0.0, 2, 0.0, 0, 48, 0, 0, 1, 5079, 879],
     [1, 0.0, 1, 0.0, 0, 19, 0, 1, 9, 1778, 1477],
     [1, 0.0, 2, 0.0, 0, 0, 0, 0, 15, 29, 543],
     [1, 0.0, 3, 0.0, 0, 77, 0, 1, 784, 526, 1235],
     [1, 0.0, 2, 0.0, 0, 81, 1, 0, 3, 9123, 6144],
     [1, 0.0, 2, 0.0, 0, 33, 0, 0, 15, 134, 416],
     [1, 0.0, 2, 0.0, 0, 79, 0, 1, 38, 506, 804],
     [1, 0.0, 2, 0.0, 0, 0, 0, 0, 20, 27, 2557],
     [1, 0.125, 2, 0.0, 0, 0, 0, 0, 15, 9, 1151],
     [1, 0.42105263157894735, 2, 0.0, 0, 0, 0, 0, 18, 12, 1212],
     [1, 0.0, 1, 0.0, 0, 0, 0, 0, 15, 14, 600],
     [1, 0.0, 5, 0.0, 0, 25, 0, 0, 12, 1224, 774],
     [1, 0.0, 1, 0.0, 0, 0, 0, 0, 15, 23, 2056],
     [1, 0.42857142857142855, 1, 0.0, 0, 0, 0, 0, 18, 27, 395],
     [1, 0.0, 2, 0.0, 0, 0, 0, 1, 10, 444, 1116],
     [1, 0.0, 1, 0.0, 0, 43, 0, 0, 57, 214, 2377],
     [1, 0.047619047619047616, 2, 0.0, 0, 0, 0, 1, 15, 15, 6047],
     [1, 0.05263157894736842, 2, 0.0, 0, 1, 0, 0, 15, 55, 5313],
     [1, 0.18181818181818182, 2, 0.0, 0, 0, 0, 0, 16, 95, 1228],
     [1, 0.15384615384615385, 1, 0.0, 0, 0, 0, 0, 16, 56, 3665],
     [1, 0.0, 1, 0, 0, 0, 0, 0, 15, 5, 1568],
     [0, 0.16666666666666666, 2, 0.0, 0, 0, 0, 1, 3, 8, 28],
     [1, 0.4117647058823529, 2, 0.0, 0, 0, 0, 0, 1, 69, 196]]




```python
# Generate pandas dataframe 
y_test_data = pd.DataFrame(y_table,
                         columns = ['profile pic', 
                                    'nums/length username', 
                                    'fullname words',
                                    'nums/length fullname',
                                    'name==username',
                                    'description length',
                                    'external URL',
                                    'private',
                                    '#posts',
                                    '#followers',
                                    '#follows'])
y_test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.142857</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>32</td>
      <td>1549</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.222222</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>15</td>
      <td>337</td>
      <td>2058</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.250000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>310</td>
      <td>6343</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
      <td>97</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>14107</td>
      <td>7514</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.363636</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>8</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.250000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>87</td>
      <td>6741</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>21</td>
      <td>24</td>
      <td>5862</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>27</td>
      <td>1289</td>
      <td>689</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>31</td>
      <td>148</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>119</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>636</td>
      <td>1293</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>144</td>
      <td>3617</td>
      <td>1346</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.214286</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>71</td>
      <td>7495</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0.133333</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>113</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>305</td>
      <td>303</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0.444444</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>283</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>115</td>
      <td>7506</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0.062500</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>272</td>
      <td>1446</td>
      <td>2362</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0.153846</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1150</td>
      <td>732</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>60</td>
      <td>1631</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>11</td>
      <td>221</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>21</td>
      <td>23</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0.230769</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>173</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>0.250000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>457</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0.500000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>831</td>
      <td>5424</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>150</td>
      <td>1</td>
      <td>0</td>
      <td>158</td>
      <td>7063</td>
      <td>1355</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>15</td>
      <td>39</td>
      <td>2045</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.055556</td>
      <td>0</td>
      <td>127</td>
      <td>0</td>
      <td>0</td>
      <td>196</td>
      <td>486</td>
      <td>198</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>76</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>509</td>
      <td>372</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5079</td>
      <td>879</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>1778</td>
      <td>1477</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>29</td>
      <td>543</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>77</td>
      <td>0</td>
      <td>1</td>
      <td>784</td>
      <td>526</td>
      <td>1235</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>81</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>9123</td>
      <td>6144</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>134</td>
      <td>416</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>79</td>
      <td>0</td>
      <td>1</td>
      <td>38</td>
      <td>506</td>
      <td>804</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>27</td>
      <td>2557</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0.125000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>9</td>
      <td>1151</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>0.421053</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>12</td>
      <td>1212</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>14</td>
      <td>600</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>0.000000</td>
      <td>5</td>
      <td>0.000000</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1224</td>
      <td>774</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>23</td>
      <td>2056</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>0.428571</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>27</td>
      <td>395</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>444</td>
      <td>1116</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>214</td>
      <td>2377</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>0.047619</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>15</td>
      <td>15</td>
      <td>6047</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>0.052632</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>55</td>
      <td>5313</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1</td>
      <td>0.181818</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>95</td>
      <td>1228</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1</td>
      <td>0.153846</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>56</td>
      <td>3665</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>5</td>
      <td>1568</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0</td>
      <td>0.166667</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>28</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>0.411765</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>69</td>
      <td>196</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Predict (no retraining!)
rfc_labels_y = rfc_model.predict(y_test_data)
rfc_labels_y
```




    array([1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
           1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
           1, 1, 1, 1, 1, 1])




```python
# Calculate the number of fake accounts in the random sample of 50 followers
no_fakes_y = len([x for x in rfc_labels_y if x==1])
```


```python
# Calculate the authenticity
y_authenticity = (len(y_random_followers) - no_fakes_y) * 100 / len(y_random_followers)
print("User Y's Instagram Followers is " + str(y_authenticity) + "% authentic.")
```

    User Y's Instagram Followers is 38.0% authentic.


Ahh, the joys of being right!

**USER Y LIKES ANALYSIS**


```python
y_posts = get_user_posts(userID_y, 10)
```

    Total posts retrieved: 18



```python
y_random_post = random.sample(y_posts, 1)
```


```python
y_likers = api.media_likers(y_random_post[0]['id'])
```


```python
y_likers_usernames = [liker['username'] for liker in y_likers['users']]
```


```python
y_random_likers = random.sample(y_likers_usernames, 50)
```


```python
y_likers_infos = []

for liker in y_random_likers:
    info = api.user_info(get_ID(liker))['user']
    y_likers_infos.append(info)
```


```python
y_likers_table = []

for info in y_likers_infos:
    y_likers_table.append(get_data(info))
    
y_likers_table
```




    [[1, 0.0, 2, 0.0, 0, 0, 0, 0, 2, 897, 830],
     [0, 0.0, 2, 0.0, 0, 0, 0, 1, 0, 129, 132],
     [1, 0.0, 2, 0.0, 0, 8, 0, 1, 72, 1157, 698],
     [1, 0.0, 1, 0, 0, 10, 0, 1, 6, 1410, 619],
     [1, 0.0, 1, 0.0, 0, 0, 0, 0, 0, 1916, 731],
     [1, 0.2222222222222222, 3, 0.0, 0, 72, 0, 1, 13, 950, 649],
     [1, 0.0, 1, 0.0, 0, 19, 0, 1, 17, 1543, 1289],
     [1, 0.2, 5, 0.0, 0, 11, 0, 0, 33, 1076, 606],
     [1, 0.0, 1, 0.0, 0, 104, 0, 1, 6, 202, 485],
     [1, 0.2, 1, 0.0, 0, 15, 0, 0, 7, 1262, 679],
     [1, 0.15384615384615385, 2, 0.0, 0, 0, 0, 0, 6, 1150, 732],
     [1, 0.0, 1, 0.0, 0, 17, 1, 0, 28, 2442, 629],
     [1, 0.0, 2, 0.0, 0, 61, 0, 0, 159, 556, 765],
     [1, 0.0, 2, 0.0, 0, 34, 0, 1, 10, 531, 526],
     [1, 0.0, 3, 0.0, 0, 127, 0, 0, 23, 1137, 909],
     [1, 0.0, 2, 0.0, 0, 66, 0, 1, 25, 583, 805],
     [1, 0.13333333333333333, 2, 0.0, 0, 67, 1, 0, 141, 4615, 1948],
     [1, 0.0, 2, 0.0, 0, 47, 0, 1, 387, 75, 162],
     [1, 0.0, 1, 0.0, 0, 142, 0, 1, 8144, 664, 1527],
     [1, 0.0, 3, 0.0, 0, 4, 0, 1, 1, 466, 325],
     [1, 0.058823529411764705, 1, 0.0, 0, 32, 0, 0, 14, 419, 414],
     [1, 0.0, 3, 0.0, 0, 75, 1, 0, 353, 1399, 764],
     [1, 0.0, 1, 0, 0, 0, 0, 0, 9, 611, 554],
     [1, 0.0, 1, 0.0, 0, 29, 0, 1, 3, 2064, 1077],
     [1, 0.0, 1, 0.0, 0, 26, 0, 1, 37, 628, 714],
     [1, 0.0, 2, 0.0, 0, 89, 1, 1, 243, 2316, 1030],
     [1, 0.0, 2, 0.0, 0, 140, 1, 0, 666, 4460, 492],
     [1, 0.0, 2, 0.0, 0, 20, 0, 0, 71, 4101, 878],
     [1, 0.0, 2, 0.0, 0, 5, 0, 0, 148, 424, 716],
     [1, 0.0, 1, 0, 0, 0, 0, 1, 2, 640, 730],
     [1, 0.0, 2, 0.0, 0, 64, 0, 1, 8, 1141, 891],
     [1, 0.0, 3, 0.0, 0, 29, 0, 1, 10, 1378, 986],
     [1, 0.0, 2, 0.0, 0, 14, 0, 1, 3, 994, 698],
     [1, 0.0, 1, 0.0, 0, 29, 0, 1, 43, 181, 169],
     [1, 0.0, 1, 0.0, 0, 58, 1, 0, 24, 1144, 1091],
     [1, 0.0, 2, 0.0, 0, 25, 0, 1, 36, 687, 574],
     [1, 0.0, 3, 0.0, 0, 8, 0, 1, 33, 1846, 996],
     [1, 0.5714285714285714, 2, 0.0, 0, 18, 0, 1, 202, 1180, 600],
     [1, 0.0, 2, 0.0, 0, 7, 0, 0, 45, 1206, 676],
     [1, 0.0, 2, 0.0, 0, 76, 0, 0, 12, 661, 3004],
     [1, 0.0, 1, 0.0, 0, 9, 0, 1, 5, 759, 706],
     [0, 0.0, 3, 0.0, 0, 61, 0, 1, 9, 439, 612],
     [1, 0.16666666666666666, 1, 0.0, 0, 0, 0, 1, 3, 911, 822],
     [1, 0.4, 2, 0.0, 0, 82, 0, 0, 99, 556, 733],
     [1, 0.0, 2, 0.0, 0, 80, 0, 1, 21, 478, 385],
     [1, 0.0, 1, 0, 0, 0, 0, 1, 0, 653, 312],
     [1, 0.0, 1, 0.0, 0, 13, 0, 1, 40, 713, 657],
     [1, 0.0, 2, 0.0, 0, 0, 0, 1, 4, 113, 311],
     [1, 0.0, 2, 0.0, 0, 33, 0, 0, 74, 3564, 1051],
     [1, 0.0, 1, 0.0, 0, 121, 0, 0, 958, 904, 479]]




```python
y_likers_data = pd.DataFrame(y_likers_table,
                         columns = ['profile pic', 
                                    'nums/length username', 
                                    'fullname words',
                                    'nums/length fullname',
                                    'name==username',
                                    'description length',
                                    'external URL',
                                    'private',
                                    '#posts',
                                    '#followers',
                                    '#follows'])
y_likers_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>profile pic</th>
      <th>nums/length username</th>
      <th>fullname words</th>
      <th>nums/length fullname</th>
      <th>name==username</th>
      <th>description length</th>
      <th>external URL</th>
      <th>private</th>
      <th>#posts</th>
      <th>#followers</th>
      <th>#follows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>897</td>
      <td>830</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>129</td>
      <td>132</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>72</td>
      <td>1157</td>
      <td>698</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1410</td>
      <td>619</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1916</td>
      <td>731</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.222222</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
      <td>950</td>
      <td>649</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>1543</td>
      <td>1289</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.200000</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>1076</td>
      <td>606</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>104</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>202</td>
      <td>485</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.200000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1262</td>
      <td>679</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.153846</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1150</td>
      <td>732</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>28</td>
      <td>2442</td>
      <td>629</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>556</td>
      <td>765</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>531</td>
      <td>526</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>127</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>1137</td>
      <td>909</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>66</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>583</td>
      <td>805</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0.133333</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>67</td>
      <td>1</td>
      <td>0</td>
      <td>141</td>
      <td>4615</td>
      <td>1948</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>1</td>
      <td>387</td>
      <td>75</td>
      <td>162</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
      <td>1</td>
      <td>8144</td>
      <td>664</td>
      <td>1527</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>466</td>
      <td>325</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0.058824</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>419</td>
      <td>414</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>75</td>
      <td>1</td>
      <td>0</td>
      <td>353</td>
      <td>1399</td>
      <td>764</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>611</td>
      <td>554</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2064</td>
      <td>1077</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>1</td>
      <td>37</td>
      <td>628</td>
      <td>714</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>243</td>
      <td>2316</td>
      <td>1030</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>140</td>
      <td>1</td>
      <td>0</td>
      <td>666</td>
      <td>4460</td>
      <td>492</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>71</td>
      <td>4101</td>
      <td>878</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>148</td>
      <td>424</td>
      <td>716</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>640</td>
      <td>730</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>1141</td>
      <td>891</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>1378</td>
      <td>986</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>994</td>
      <td>698</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>43</td>
      <td>181</td>
      <td>169</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>58</td>
      <td>1</td>
      <td>0</td>
      <td>24</td>
      <td>1144</td>
      <td>1091</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>36</td>
      <td>687</td>
      <td>574</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>33</td>
      <td>1846</td>
      <td>996</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>0.571429</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>202</td>
      <td>1180</td>
      <td>600</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>1206</td>
      <td>676</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>661</td>
      <td>3004</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>759</td>
      <td>706</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>439</td>
      <td>612</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>0.166667</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>911</td>
      <td>822</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>0.400000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>556</td>
      <td>733</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>1</td>
      <td>21</td>
      <td>478</td>
      <td>385</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>653</td>
      <td>312</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>713</td>
      <td>657</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>113</td>
      <td>311</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>74</td>
      <td>3564</td>
      <td>1051</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>121</td>
      <td>0</td>
      <td>0</td>
      <td>958</td>
      <td>904</td>
      <td>479</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Predict!
y_likers_pred = rfc_model.predict(y_likers_data)
y_likers_pred
```




    array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0])




```python
# Calculate the number of fake likes
no_fakes_yl = len([x for x in y_likers_pred if x==1])

# Calculate media likes authenticity
y_post_authenticity = (len(y_random_likers) - no_fakes_yl) * 100 / len(y_random_likers)
print("The media with the ID:YYYYY has " + str(y_post_authenticity) + "% authentic likes.")
```

    The media with the ID:YYYYY has 96.0% authentic likes.


Very high likes authenticity but very low follower authenticity? How is that possible?

We can use **engagement rates** to explain this phenomena further.

Engagement rate = average number of engagements (likes+comments) / number of followers)


```python
y_posts[0].keys()
```




    dict_keys(['taken_at', 'pk', 'id', 'device_timestamp', 'media_type', 'code', 'client_cache_key', 'filter_type', 'carousel_media_count', 'carousel_media', 'can_see_insights_as_brand', 'location', 'lat', 'lng', 'user', 'can_viewer_reshare', 'caption_is_edited', 'comment_likes_enabled', 'comment_threading_enabled', 'has_more_comments', 'max_num_visible_preview_comments', 'preview_comments', 'can_view_more_preview_comments', 'comment_count', 'inline_composer_display_condition', 'inline_composer_imp_trigger_time', 'like_count', 'has_liked', 'top_likers', 'photo_of_you', 'caption', 'can_viewer_save', 'organic_tracking_token'])




```python
count = 0

for post in y_posts:
    count += post['comment_count']
    count += post['like_count']
    
average_engagements = count / len(y_posts)
engagement_rate = average_engagements*100 / len(y_followers)

engagement_rate
```




    9.50268408791654



This means that only roughly 9.5% of user Y's followers engage with their content. 

## Part 8: Thoughts

**Making sense of the result**

So user X received an 82% follower authenticity score and a 92% media likes authenticity on one of their posts. Is that good enough? What about user Y with a 35% follower authenticity score and a 96% media likes authenticity?

Since this entire notebook is an exploratory analysis, there's not really a hard line between a 'good' influencer and a 'bad' influencer. For user X, we can tell that the user has authentic and loyal followers. However for user Y, we can assume that they have a rather low authentic follower score, however their likes consist of real followers. This means that user Y might have invested on buying followers, but not likes! This causes a really low engagement rate.

In fact, with a little bit more research, you can sort of establish a pattern just by observation:
- High follower authenticity, high media authenticity, high engagement rate = authentic user
- Low follower authenticity, high media authenticity, low engagement rate = buys followers, does not buy likes
- Low follower authenticity, high media authenticity, high engagement rate = buys followers AND likes
- ... and so on!

**So is this influencer worth investing or not?**

Remember that we used a *random sample* of 50 followers out of thousands. As objective as random sampling could be, it still isn't an *absolutely complete* picture of the user's followers. However, the follower authenticity combined with the media likes authenticity still provides an insight for brands who are planning to invest on the influencer. 

Personally, I feel like any number under 50% is rather suspicious, and there are other ways that you can confirm this suspicion:
- Low engagement rates (engagement rate = average number of engagements (likes+comments) / number of followers)
- Spikes in follower growth (uneven growth chart)
- Comments (loyal followers acutally care about the user's content)

But of course, you have to be aware of tech-savvy influencers who cheats the audit system and try to avoid getting caught, such as influencers who buys 'drip-followers' - i.e. you buy followers in bulk but they arrive slowly. This method will make their follower growth seem gradual.

**Conclusion**

The rapid growth of technology allows anyone with a computer to create bots to follow users and like media on any platform. However, this also means that our ability to detect fake engagements should also improve!

Businesses, small or large, invest on social media influencers to reach a wider audience, especially during times of a global pandemic where everyone is constantly on their phones! Less tech-savvy and less aware ones are prone to this kind of misinformation.

For brands who rely on influencers for marketing, it is highly recommended to check out services such as SocialBlade to check user authenticity and engagement. Some services are more pricey, but is definitely worth the investment!

