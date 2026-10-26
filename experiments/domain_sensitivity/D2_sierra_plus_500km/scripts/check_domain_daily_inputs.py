import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("FSW_ORIG_REPO", "/global/homes/h/hyvchen/foreswe-gpu-optimization")
os.environ.setdefault("FSW_DOMAIN_ROOT", os.path.join(os.getenv("FSW_RUN_ROOT", "/pscratch/sd/h/hyvchen/foreswe-gpu-optimization/runs"), "D2_sierra_plus_500km"))
os.environ.setdefault("FSW_DATA_DIR", os.path.join(os.environ["FSW_DOMAIN_ROOT"], "data_subset"))
os.environ.setdefault("FSW_OUTPUTS", os.path.join(os.environ["FSW_DOMAIN_ROOT"], "outputs"))
os.environ.setdefault("FSW_RESULTS_ACTUAL", os.path.join(os.environ["FSW_DOMAIN_ROOT"], "results", "actual"))
os.environ.setdefault("FSW_RESULTS_PREDICT", os.path.join(os.environ["FSW_DOMAIN_ROOT"], "results", "predict"))
os.environ.setdefault("FSW_CHECKPOINTS", os.path.join(os.environ["FSW_DOMAIN_ROOT"], "checkpoints"))
os.environ.setdefault("FSW_STATION_SUBSET_CSV", os.path.join(os.environ["FSW_DOMAIN_ROOT"], "station_subset.csv"))

import extended_dl_forecasting as DL


def main():
    data_dir = os.environ["FSW_DATA_DIR"]
    outputs_dir = os.environ["FSW_OUTPUTS"]
    station_subset_csv = os.environ["FSW_STATION_SUBSET_CSV"]
    station_subset = pd.read_csv(station_subset_csv)
    expected_stations = len(station_subset)

    print("Active data directory", data_dir)
    print("Active output directory", outputs_dir)
    print("Active station subset CSV", station_subset_csv)
    print("Expected station count from station_subset.csv", expected_stations)
    print("Sierra eval station count", int(station_subset["is_sierra_eval_station"].astype(bool).sum()))

    dataloader = DL.DataLoader(test_indices=[], temp_window=7)
    all_locations_distances, all_locations_angularities = dataloader.getDistanceAndAngularity()

    dataloader.cleanDataJunk()
    dataloader.normalizeData()
    dataloader.addDayInfo()
    dataloader.prepareInputsAndOutputs()

    model_inputs, model_outputs = dataloader.getModelInputsAndOutputs()
    dataloader.completeInputsAndOutputs()

    n_stations = len(dataloader.snotel_locations_info)
    test_indices = dataloader.getTestingIndices()

    print("N_STATIONS", n_stations)
    print("Metadata shape", dataloader.snotel_locations_info.shape)
    print("SWE CSV shape", dataloader.swe_values.shape)
    print("Model input shape", model_inputs.shape)
    print("Model output shape", model_outputs.shape)
    print("Distance matrix shape", all_locations_distances.shape)
    print("Angularity matrix shape", all_locations_angularities.shape)
    print("Deterministic subset test station count", len(test_indices))
    print("Deterministic subset test station indices", test_indices)

    if int(station_subset["is_sierra_eval_station"].astype(bool).sum()) != 43:
        raise ValueError("Expected all 43 Sierra eval stations to be present in station_subset.csv")

    if n_stations != expected_stations:
        raise ValueError("Expected {} stations but found {}".format(expected_stations, n_stations))

    if dataloader.swe_values.shape[1] != expected_stations + 1:
        raise ValueError("Expected SWE CSV to have {} columns including Date but found {}".format(expected_stations + 1, dataloader.swe_values.shape[1]))

    if model_inputs.shape[1] != expected_stations:
        raise ValueError("Model input station dimension mismatch: {}".format(model_inputs.shape))

    if model_outputs.shape[1] != expected_stations:
        raise ValueError("Model output station dimension mismatch: {}".format(model_outputs.shape))

    if all_locations_distances.shape != (expected_stations, expected_stations):
        raise ValueError("Distance matrix shape mismatch: {}".format(all_locations_distances.shape))

    if all_locations_angularities.shape != (expected_stations, expected_stations):
        raise ValueError("Angularity matrix shape mismatch: {}".format(all_locations_angularities.shape))

    if sorted(test_indices) != list(test_indices):
        raise ValueError("Expected deterministic test indices to be sorted: {}".format(test_indices))

    required_files = [
        "all_stations_day_length.csv","precipitation_info.csv","rain_info.csv","rmax_info.csv","rmin_info.csv",
        "snow_info.csv","sph_info.csv","srad_info.csv","swe_values.csv","tb_19_info.csv","tb_37_info.csv",
        "tb_diff_info.csv","tmax_info.csv","tmin_info.csv","windspeed_info.csv"
    ]
    for fname in required_files:
        shape = pd.read_csv(os.path.join(data_dir, fname), nrows=5).shape[1]
        if shape != expected_stations + 1:
            raise ValueError("{} should have Date + {} station columns but has {} columns".format(fname, expected_stations, shape))

    print("Domain daily input shape check passed.")


if __name__ == "__main__":
    main()
