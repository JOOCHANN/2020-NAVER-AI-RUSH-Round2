# Anomaly Detection

Repo for training a baseline model for the AI Rush anomaly detection challenge.
The baseline model is a one hidden layer MLP.

## Important notes
- The function that's bind to NSML infer needs to output a pandas dataframe
- The dataframe should contain `mlFlag` column which consists of 0's and 1's.

## Run experiment

To run the baseline model training, run
```
nsml run -d airush_anomaly_0 -m "a message" 
```

## Metric

We use F1 score.

## Data

- The dataset consists of 8 categorical variables and 2 continuous variables.
- 8 categorical variables consists of **{'accessCountryCode', 'awsIp', 'blackIp', 'joinCountryCode', 'logType', 'mobileIp', 'proxyIp', 'vpnIp'}**
- 2 continuous variables consists of **{'px', 'py'}**
- All the columns may contain NaNs.
- You may encounter unseen category in test set.

You can find the detailed description of each column below

```
accessCountryCode: KR, JP, CH ...
awsIp            : 0, 1
blackIp          : 0, 1
joinCountryCode  : KR, JP, CH ...
logType          : click, comment ... etc.
mobileIp         : 0, 1
proxyIp          : 0, 1
vpnIp            : 0, 1
px (cont.)       : x coordinate of click pointer
py (cont.)       : y coordinate of click pointer
```
