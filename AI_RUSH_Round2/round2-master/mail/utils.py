import numpy as np


def read_mail_data(data_path, label_path, max_title, max_content):
    mail_data = []
    with open(data_path, "r", encoding="utf-8") as data_file:
        read_data = data_file.readlines()
        for line in read_data:
            parse_line = line.strip().split("\t")
            title = [int(i) for i in parse_line[0].split(",")[:max_title]]
            content = [int(i) for i in parse_line[1].split(",")[:max_content]]
            mail_data.append((title, content))

    mail_label = []
    with open(label_path, "r", encoding="utf-8") as label_file:
        read_label = label_file.readlines()
        for line in read_label:
            label = int(line.strip())
            mail_label.append(label)

    mail_dataset = []
    for label, (title, content) in zip(mail_label, mail_data):
        mail_dataset.append((label, title, content))

    return mail_dataset


def read_mail_data_legacy(data_path, max_title, max_content):
    mail_data = []
    with open(data_path, "r", encoding="utf-8") as data:
        read_data = data.readlines()
        for line in read_data:
            parse_line = line.strip().split("\t")
            label = 1 if parse_line[0] == "spam" else 0
            title = [int(i) for i in parse_line[1].split(",")[:max_title]]
            content = [int(i) for i in parse_line[2].split(",")[:max_content]]
            mail_data.append((label, title, content))
    return mail_data


def roc_metric(results, threshold=0.5):
    roc = np.zeros((2, 2), dtype=np.float32)
    for label, score in results:
        if label < 0.5:
            if score < threshold:
                roc[0][0] += 1.0
            else:
                roc[0][1] += 1.0
        else:
            if score < threshold:
                roc[1][0] += 1.0
            else:
                roc[1][1] += 1.0
    return roc


def calculate_f1(roc):
    precision = roc[0][0] / (roc[0][0] + roc[1][0] + 1e-4)
    recall = roc[0][0] / (roc[0][0] + roc[0][1] + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return [precision, recall, f1]
