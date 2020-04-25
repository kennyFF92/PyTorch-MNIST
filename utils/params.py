import json
from datetime import datetime

default_params = {
    "random_seed": 23,
    "use_cuda": False,
    "input_size": [28, 28],
    "classes": 10,
    "val_perc": 0.2,
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "data_path": "./data",
    "tensorboard_path": "./runs",
    "trained_model_path": "./model"
}


def parse_parameters(json_path="./parameters.json"):
    with open(json_path) as json_file:
        params = json.load(json_file)

    # Fill params with default if missing
    for key, value in default_params.items():
        if key not in params:
            params[key] = value

    return params


def get_start_time():
    start_time = str(datetime.now()).replace(' ',
                                             '_').replace(':',
                                                          '-').split('.')[0]
    return start_time
