# q2
import os
import json
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


def evaluate(prediction_fname, ground_truth_fname):
    dataset_name = 'q2'
    label_types = {'q1': 'station_name',
                   'q2': 'mood_tag',
                   'q3': 'genre'}
    with open(prediction_fname) as f:
        predictions = json.load(f)
        assert len(predictions) in [9, 24, 49, 98]

    with open(os.path.join(ground_truth_fname)) as f:
        track_info = json.load(f)
        label_type = label_types[dataset_name]
        label_dict = track_info[label_type]
        labels = [label_dict[str(idx)] for idx in range(len(label_dict))]
        assert len(labels) in [9, 24, 49, 98]

    assert len(predictions) == len(labels)

    mlb = MultiLabelBinarizer()
    mlb.fit(labels + predictions)
    labels = mlb.transform(labels)
    predictions = mlb.transform(predictions)

    score = f1_score(labels, predictions, average='micro')
    return score


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally
    # prediction file requires type casting because '\n' character can be contained.
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str)
    config = args.parse_args()
    # print the evaluation result
    # evaluation prints only int or float value.
    print(evaluate(config.prediction, config.test_label_path))

