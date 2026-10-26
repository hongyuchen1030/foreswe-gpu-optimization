import os
import sys

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

ROOT = os.environ.setdefault(
    "FSW_WEEKLY_ROOT",
    "/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/original_weekly",
)
os.environ.setdefault("BASE_URL", ROOT)
os.environ.setdefault("FSW_DATA_DIR", os.path.join(ROOT, "data"))
os.environ.setdefault("FSW_OUTPUTS", os.path.join(ROOT, "outputs"))
os.environ.setdefault("FSW_RESULTS_ACTUAL", os.path.join(ROOT, "results", "actual"))
os.environ.setdefault("FSW_RESULTS_PREDICT", os.path.join(ROOT, "results", "predict"))
os.environ.setdefault("FSW_CHECKPOINTS", os.path.join(ROOT, "checkpoints"))
os.environ.setdefault("HF_HOME", "/pscratch/sd/h/hyvchen/foreswe_model_cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/pscratch/sd/h/hyvchen/foreswe_model_cache/huggingface/hub")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/pscratch/sd/h/hyvchen/foreswe_model_cache/sentence_transformers")

import extended_dl_forecasting as DL


def main():
    data_dir = os.environ["FSW_DATA_DIR"]
    results_actual_dir = os.environ["FSW_RESULTS_ACTUAL"]
    results_predict_dir = os.environ["FSW_RESULTS_PREDICT"]
    outputs_dir = os.environ["FSW_OUTPUTS"]

    metadata = pd.read_csv(os.path.join(data_dir, "snotel_location_info_2.csv"))
    n_stations = len(metadata)

    print("Active weekly root", ROOT)
    print("Active data directory", data_dir)
    print("Active outputs directory", outputs_dir)
    print("Active results actual directory", results_actual_dir)
    print("Active results predict directory", results_predict_dir)
    print("Expected weekly horizons days", [7, 14, 21, 28])
    print("N_STATIONS", n_stations)
    print("Metadata shape", metadata.shape)

    required_daily_prediction_files = [
        os.path.join(results_predict_dir, "{}_daily_predict.pkl".format(year))
        for year in range(2014, 2019)
    ]
    required_daily_actual_files = [
        os.path.join(results_actual_dir, "{}_daily_window_actual.pkl".format(year))
        for year in range(2014, 2019)
    ]

    for path in required_daily_prediction_files + required_daily_actual_files:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        print("FOUND", path)

    dataloader = DL.DataLoader(test_indices=[], temp_window=7)
    distances, angularities = dataloader.getDistanceAndAngularity()
    dataloader.cleanDataJunk()
    dataloader.normalizeData()
    dataloader.addDayInfo()
    dataloader.prepareInputsAndOutputs()
    model_inputs, model_outputs = dataloader.getModelInputsAndOutputs()
    dataloader.completeInputsAndOutputs()

    print("Model input shape", model_inputs.shape)
    print("Model output shape", model_outputs.shape)
    print("Distance matrix shape", distances.shape)
    print("Angularity matrix shape", angularities.shape)
    print("Testing year lengths", [len(v) for v in dataloader.getTestingDataIndices()])

    if model_inputs.shape[1] != n_stations:
        raise ValueError("Model input station dimension mismatch: {}".format(model_inputs.shape))
    if model_outputs.shape[1] != n_stations:
        raise ValueError("Model output station dimension mismatch: {}".format(model_outputs.shape))
    if distances.shape != (n_stations, n_stations):
        raise ValueError("Distance matrix shape mismatch: {}".format(distances.shape))
    if angularities.shape != (n_stations, n_stations):
        raise ValueError("Angularity matrix shape mismatch: {}".format(angularities.shape))

    sample_daily = np.asarray(joblib.load(required_daily_prediction_files[0]))
    sample_actual = np.asarray(joblib.load(required_daily_actual_files[0]))

    print("Sample daily prediction shape", sample_daily.shape)
    print("Sample daily actual shape", sample_actual.shape)

    if sample_daily.shape[0] != n_stations:
        raise ValueError("Daily prediction station dimension mismatch: {}".format(sample_daily.shape))
    if sample_actual.shape[1] != n_stations:
        raise ValueError("Daily actual station dimension mismatch: {}".format(sample_actual.shape))

    if sample_daily.shape[-1] != 1:
        raise ValueError("Expected final daily GP predictions to have trailing dimension 1: {}".format(sample_daily.shape))
    if sample_actual.shape[-1] != 10:
        raise ValueError("Expected daily actual windows to have 10-day horizon dimension: {}".format(sample_actual.shape))

    print("Weekly pipeline dry check passed. The weekly model can use the paper-native 28-day horizon as step 4.")


if __name__ == "__main__":
    main()
