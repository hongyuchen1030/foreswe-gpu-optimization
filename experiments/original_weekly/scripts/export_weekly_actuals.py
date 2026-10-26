import os
import sys

import joblib
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

ROOT = os.environ.setdefault(
    "FSW_WEEKLY_ROOT",
    "/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/original_weekly",
)
os.environ.setdefault("BASE_URL", ROOT)
os.environ.setdefault("FSW_DATA_DIR", os.path.join(ROOT, "data"))
os.environ.setdefault("FSW_RESULTS_ACTUAL", os.path.join(ROOT, "results", "actual"))

import extended_dl_forecasting as DL


FORECASTING_WINDOW = 4
FORECASTING_STEP_SIZE = 7


def get_outputs(model_outputs, index, loc_indices):
    indices = sorted([index + (value * FORECASTING_STEP_SIZE) for value in range(1, FORECASTING_WINDOW + 1)])
    return model_outputs[:, loc_indices, :][indices, :, :].permute(1, 0, 2).reshape(len(loc_indices), -1)


def build_weekly_window_actual(model_outputs, data_indices, loc_indices):
    actual_outputs = []
    for input_index in data_indices:
        instance_output = get_outputs(model_outputs, input_index, loc_indices)
        actual_outputs.append(instance_output.reshape(1, -1, FORECASTING_WINDOW).cpu().numpy())
    return np.array(actual_outputs).reshape(-1, len(loc_indices), FORECASTING_WINDOW)


def main():
    results_actual_dir = os.environ["FSW_RESULTS_ACTUAL"]
    os.makedirs(results_actual_dir, exist_ok=True)

    dataloader = DL.DataLoader(test_indices=[], temp_window=7)
    dataloader.cleanDataJunk()
    dataloader.normalizeData()
    dataloader.addDayInfo()
    dataloader.prepareInputsAndOutputs()
    _, model_outputs_np = dataloader.getModelInputsAndOutputs()
    dataloader.completeInputsAndOutputs()

    model_outputs = torch.from_numpy(model_outputs_np).to(dtype=torch.float32)
    train_loc_indices = dataloader.getTrainingIndices()
    test_indices = dataloader.getTestingIndices()
    all_indices = sorted(list(train_loc_indices) + list(test_indices))

    test_years = [2014, 2015, 2016, 2017, 2018]
    for year, year_indices in zip(test_years, dataloader.getTestingDataIndices()):
        weekly_window_actual = build_weekly_window_actual(model_outputs, year_indices, all_indices)
        out_path = os.path.join(results_actual_dir, f"{year}_weekly_window_actual.pkl")
        joblib.dump(weekly_window_actual, out_path)
        print(year, weekly_window_actual.shape, out_path)

    print("Weekly actual export complete.")


if __name__ == "__main__":
    main()
