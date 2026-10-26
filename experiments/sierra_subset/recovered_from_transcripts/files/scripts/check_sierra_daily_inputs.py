
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("FSW_ORIG_REPO", "/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting")
os.environ.setdefault("FSW_SIERRA_ROOT", "/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station")
os.environ.setdefault("FSW_DATA_DIR", os.path.join(os.environ["FSW_SIERRA_ROOT"], "data_subset"))
os.environ.setdefault("FSW_OUTPUTS", os.path.join(os.environ["FSW_SIERRA_ROOT"], "outputs"))
os.environ.setdefault("FSW_RESULTS_ACTUAL", os.path.join(os.environ["FSW_SIERRA_ROOT"], "results", "actual"))
os.environ.setdefault("FSW_RESULTS_PREDICT", os.path.join(os.environ["FSW_SIERRA_ROOT"], "results", "predict"))
os.environ.setdefault("FSW_CHECKPOINTS", os.path.join(os.environ["FSW_SIERRA_ROOT"], "checkpoints"))
os.environ.setdefault("FSW_EXPECTED_STATIONS", "43")

import extended_dl_forecasting as DL


def main():
    expected_stations = int(os.environ["FSW_EXPECTED_STATIONS"])
    data_dir = os.environ["FSW_DATA_DIR"]
    outputs_dir = os.environ["FSW_OUTPUTS"]

    print("Active data directory", data_dir)
    print("Active output directory", outputs_dir)

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

    print("Sierra daily input shape check passed.")


if __name__ == "__main__":
    main()
