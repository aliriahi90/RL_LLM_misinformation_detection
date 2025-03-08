import json
import pandas as pd
import os
from sklearn.metrics import classification_report

def write_json(data, path):
    """
            Write json file
    Args:
        param data
        param path to save json file
    Return:
        json file
    """
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)
        

def load_datasets(path, dataset_name):
    if dataset_name == "FakeHealth":
        train = pd.read_csv(os.path.join(path, "FakeHealth_train.csv"))
        test = pd.read_csv(os.path.join(path, "FakeHealth_test.csv"))
    else:
        train = pd.read_csv(os.path.join(path, "ReCOVery_train.csv"))
        test = pd.read_csv(os.path.join(path, "ReCOVery_test.csv"))
    return train, test


def output_processor(predicts):
    processed_outputs = []
    for predict in predicts:
        if 'real' in predict.lower():
            processed_outputs.append('real')
        elif 'fake' in predict.lower() or "false" in predict.lower() or "not true" in predict.lower():
            processed_outputs.append('fake')
        else:
            processed_outputs.append("real")
    return processed_outputs


def evaluation_report(y_true, y_pred):
    clf_report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    return clf_report
