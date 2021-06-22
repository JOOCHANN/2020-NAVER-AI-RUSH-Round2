import os
import nsml
import json


def feed_infer(output_file, infer_func):
    result = infer_func(os.path.join(nsml.DATASET_PATH, 'test'))

    print('write output')
    with open(output_file, 'w') as f:
        json.dump(result, f)

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')

