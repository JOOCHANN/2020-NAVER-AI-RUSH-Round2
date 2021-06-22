import numpy as np

import pandas as pd

from sklearn.metrics import f1_score


def read_prediction(prediction_file):
    pred_series = pd.read_csv(prediction_file, index_col=0)['mlFlag']
    pred_array = pred_series.values.flatten()
    return pred_array


def read_ground_truth(ground_truth_file):
    gt_series = pd.read_csv(ground_truth_file, index_col=0)['mlFlag']
    gt_array = gt_series.values.flatten()
    return gt_array


# recall
def evaluate(prediction, ground_truth):
    return f1_score(ground_truth, prediction)

# user-defined function for evaluation metrics
def evaluation_metrics(prediction_file: str, ground_truth_file: str):
    # read prediction and ground truth from file
    prediction = read_prediction(prediction_file)  # NOTE: prediction is text
    ground_truth = read_ground_truth(ground_truth_file)
    return evaluate(prediction, ground_truth)


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
    # prediction file requires type casting because '\n' character can be contained.
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str)
    config = args.parse_args()
    # print the evaluation result
    # evaluation prints only int or float value.
    print(evaluation_metrics(config.prediction, config.test_label_path))
