import os
import nsml
import json


def feed_infer(output_file, infer_func):
    result = infer_func(os.path.join(nsml.DATASET_PATH, 'test'))

    with open(output_file, 'w') as f:
        json.dump(result, f)
    assert len(result) in [9, 24, 49, 98]

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')

