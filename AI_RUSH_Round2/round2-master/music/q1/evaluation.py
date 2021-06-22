import os
import json
import argparse


def evaluate(prediction_fname, ground_truth_fname):
    dataset_name = 'q1'
    label_types = {'q1': 'station_name',
                   'q2': 'mood_tag',
                   'q3': 'genre'}
    with open(prediction_fname) as f:
        predictions = json.load(f)
    with open(os.path.join(ground_truth_fname)) as f:
        track_info = json.load(f)
        label_dict = track_info[label_types[dataset_name]]
        labels = [label_dict[str(idx)] for idx in range(len(label_dict))]
    assert len(predictions) == len(labels)

    cnt = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            cnt += 1
    # print(cnt / len(labels))
    top1acc = cnt / len(labels)
    return top1acc


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
    # prediction file requires type casting because '\n' character can be contained.
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str)
    config = args.parse_args()
    # print the evaluation result
    # evaluation prints only int or float value.
    print(evaluate(config.prediction, config.test_label_path))

