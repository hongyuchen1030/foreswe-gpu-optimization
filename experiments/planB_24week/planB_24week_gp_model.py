import os
import random
import traceback

import gpytorch
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gpytorch.models import ApproximateGP
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float64)

PLANB_ROOT = os.getenv(
    "FSW_PLANB_ROOT",
    os.path.join(os.getenv("FSW_RUN_ROOT", "/pscratch/sd/h/hyvchen/foreswe-gpu-optimization/runs"), "planB_24week"),
)
ORIG_REPO_ROOT = os.getenv(
    "FSW_ORIG_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
DATA_DIR = os.getenv("FSW_DATA_DIR", os.path.join(ORIG_REPO_ROOT, "data"))
OUTPUTS_DIR = os.path.join(PLANB_ROOT, "outputs")
RESULTS_ACTUAL_DIR = os.path.join(PLANB_ROOT, "results", "actual")
RESULTS_PREDICT_DIR = os.path.join(PLANB_ROOT, "results", "predict")

for path in [PLANB_ROOT, OUTPUTS_DIR, RESULTS_ACTUAL_DIR, RESULTS_PREDICT_DIR]:
    os.makedirs(path, exist_ok=True)

SEED = int(os.getenv("PLANB_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU is running.")
else:
    print("CPU is running.")

N_TASKS = 24
N_STATIONS = len(pd.read_csv(os.path.join(DATA_DIR, "snotel_location_info_2.csv")))
TEST_YEAR_LABELS = [2015, 2016, 2017, 2018, 2019]

PLANB_GP_SMOKE = os.getenv("PLANB_GP_SMOKE", "0") != "0"
PLANB_GP_MAX_ROWS = int(os.getenv("PLANB_GP_MAX_ROWS", "4096" if PLANB_GP_SMOKE else "0"))
PLANB_GP_EPOCHS = int(os.getenv("PLANB_GP_EPOCHS", "1" if PLANB_GP_SMOKE else "2"))
PLANB_GP_BATCH = int(os.getenv("PLANB_GP_BATCH", "512" if PLANB_GP_SMOKE else "4096"))
PLANB_GP_INDUCING = int(os.getenv("PLANB_GP_INDUCING", "128" if PLANB_GP_SMOKE else "7000"))
MAX_GP_TRAIN_POINTS = int(os.getenv("MAX_GP_TRAIN_POINTS", "0"))
GP_OUTPUT_MODE = os.getenv("GP_OUTPUT_MODE", "multioutput")
DEVICE_OVERRIDE = os.getenv("DEVICE", "")
RUN_DIAGNOSTIC = os.getenv("PLANB_GP_DIAGNOSTIC", "0") != "0"
DIAGNOSTIC_SIZES = [
    int(value.strip())
    for value in os.getenv("PLANB_GP_DIAGNOSTIC_SIZES", "1000,5000,10000").split(",")
    if value.strip()
]

if DEVICE_OVERRIDE:
    device = torch.device(DEVICE_OVERRIDE)

print("Plan B root", PLANB_ROOT)
print("Weekly forecasting horizons days", [7 * value for value in range(1, N_TASKS + 1)])
print("N_STATIONS", N_STATIONS)
print("PLANB_GP_SMOKE", PLANB_GP_SMOKE)
print("PLANB_GP_MAX_ROWS", PLANB_GP_MAX_ROWS)
print("PLANB_GP_EPOCHS", PLANB_GP_EPOCHS)
print("PLANB_GP_BATCH", PLANB_GP_BATCH)
print("PLANB_GP_INDUCING", PLANB_GP_INDUCING)
print("MAX_GP_TRAIN_POINTS", MAX_GP_TRAIN_POINTS)
print("GP_OUTPUT_MODE", GP_OUTPUT_MODE)
print("DEVICE", device)
print("RUN_DIAGNOSTIC", RUN_DIAGNOSTIC)
print("DIAGNOSTIC_SIZES", DIAGNOSTIC_SIZES)

training_inputs = torch.load(os.path.join(OUTPUTS_DIR, "forecasting_modified_weekly_att_inputs_for_gp_act.pt"))
training_outputs = torch.load(os.path.join(OUTPUTS_DIR, "forecasting_modified_weekly_att_outputs_for_gp_act.pt"))
training_predicted_outputs = torch.load(os.path.join(OUTPUTS_DIR, "forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt"))

test_inputs = [
    torch.load(os.path.join(OUTPUTS_DIR, "weekly_test{}_inputs_for_gp_act.pt".format(i)))
    for i in range(1, 6)
]
test_outputs = [
    joblib.load(os.path.join(RESULTS_ACTUAL_DIR, "{}_weekly_window_actual.pkl".format(year - 1)))
    for year in TEST_YEAR_LABELS
]


def identity_normalize(tensor, mean_val, std_val):
    return tensor


full_training_inputs = torch.cat(training_inputs, dim=0).reshape(-1, 8).contiguous()
full_training_outputs = torch.cat(training_outputs, dim=0).reshape(-1, N_TASKS).contiguous()

training_inputs = full_training_inputs
training_outputs = full_training_outputs

if MAX_GP_TRAIN_POINTS > 0:
    training_inputs = training_inputs[:MAX_GP_TRAIN_POINTS]
    training_outputs = training_outputs[:MAX_GP_TRAIN_POINTS]

if PLANB_GP_MAX_ROWS > 0:
    training_inputs = training_inputs[:PLANB_GP_MAX_ROWS]
    training_outputs = training_outputs[:PLANB_GP_MAX_ROWS]

train_mean = training_inputs.mean(dim=0, keepdim=True)
train_std = training_inputs.std(dim=0, keepdim=True)
training_inputs = identity_normalize(training_inputs, train_mean, train_std)

training_inputs = training_inputs.to(device=device, dtype=torch.float64)
training_outputs = training_outputs.to(device=device, dtype=torch.float64)


def bytes_to_gib(num_bytes):
    return num_bytes / float(1024 ** 3)


def dense_square_bytes(n_points, dtype_bytes):
    return n_points * n_points * dtype_bytes


def dense_multitask_bytes(n_points, n_tasks, dtype_bytes):
    return (n_points * n_tasks) * (n_points * n_tasks) * dtype_bytes


def tensor_summary(name, tensor):
    print(
        "{} shape={} dtype={} device={}".format(
            name, tuple(tensor.shape), tensor.dtype, tensor.device
        )
    )


class MultitaskGPModel(ApproximateGP):
    def __init__(self, num_latents=5, num_tasks=N_TASKS, n_inducing_points=PLANB_GP_INDUCING, input_dim=8):
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.n_inducing_points = n_inducing_points
        inducing_points = torch.rand(self.num_latents, self.n_inducing_points, input_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([self.num_latents])
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=self.num_tasks,
            num_latents=self.num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.num_latents])) +
            gpytorch.kernels.LinearKernel(batch_shape=torch.Size([self.num_latents])),
            batch_shape=torch.Size([self.num_latents]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SingleLeadGPModel(ApproximateGP):
    def __init__(self, n_inducing_points=PLANB_GP_INDUCING, input_dim=8):
        inducing_points = torch.rand(n_inducing_points, input_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_gp_components():
    if GP_OUTPUT_MODE == "single_lead":
        model = SingleLeadGPModel(n_inducing_points=PLANB_GP_INDUCING, input_dim=8).to(device=device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
        target = training_outputs[:, 0].contiguous()
    else:
        model = MultitaskGPModel(num_tasks=N_TASKS, num_latents=5, n_inducing_points=PLANB_GP_INDUCING, input_dim=8).to(device=device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=N_TASKS).to(device=device)
        target = training_outputs
    return model, likelihood, target


def run_training_epoch(model, likelihood, x_data, y_data):
    train_dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(train_dataset, batch_size=PLANB_GP_BATCH, shuffle=True)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=0.12,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x_data.size(0))
    model.train()
    likelihood.train()
    for epoch in range(PLANB_GP_EPOCHS):
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            print("epoch {} batch {} loss {}".format(epoch + 1, batch_index + 1, loss.item()))


def run_diagnostic():
    print("=== PLANB GP DIAGNOSTIC START ===")
    tensor_summary("train_x", training_inputs)
    tensor_summary("train_y", training_outputs)
    print("N_train_points", training_inputs.size(0))
    print("output_dim", training_outputs.size(-1))
    print("dtype_bytes", training_inputs.element_size())
    print("dense_NxN_gib", bytes_to_gib(dense_square_bytes(training_inputs.size(0), training_inputs.element_size())))
    print(
        "dense_multitask_(N*D)x(N*D)_gib",
        bytes_to_gib(dense_multitask_bytes(training_inputs.size(0), training_outputs.size(-1), training_inputs.element_size())),
    )
    for index, (x_raw, y_raw) in enumerate(zip(test_inputs, test_outputs), start=1):
        x_test = torch.tensor(x_raw.reshape(-1, 8)).contiguous().to(device=device, dtype=torch.float64)
        y_test = torch.tensor(y_raw.reshape(-1, N_TASKS)).contiguous().to(device=device, dtype=torch.float64)
        tensor_summary("test{}_x".format(index), x_test)
        tensor_summary("test{}_y".format(index), y_test)
    model, likelihood, train_target = build_gp_components()
    print("model_class", model.__class__.__name__)
    print("likelihood_class", likelihood.__class__.__name__)
    print("uses_lmc_variational_strategy", isinstance(getattr(model, "variational_strategy", None), gpytorch.variational.LMCVariationalStrategy))
    print("variational_strategy_class", model.variational_strategy.__class__.__name__)
    print("kernel_class", model.covar_module.__class__.__name__)
    print("base_kernel_class", model.covar_module.base_kernel.__class__.__name__)
    print("likelihood_num_tasks", getattr(likelihood, "num_tasks", 1))
    print("n_inducing_points", PLANB_GP_INDUCING)
    print("num_latents", getattr(model, "num_latents", 1))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for n_points in DIAGNOSTIC_SIZES:
        n_points = min(n_points, full_training_inputs.size(0))
        x_diag = full_training_inputs[:n_points].to(device=device, dtype=torch.float64)
        if GP_OUTPUT_MODE == "single_lead":
            y_diag = full_training_outputs[:n_points, 0].contiguous().to(device=device, dtype=torch.float64)
        else:
            y_diag = full_training_outputs[:n_points].contiguous().to(device=device, dtype=torch.float64)
        diag_model, diag_likelihood, _ = build_gp_components()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            run_training_epoch(diag_model, diag_likelihood, x_diag, y_diag)
            peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() and device.type == "cuda" else 0
            print("diagnostic_n={} status=ok peak_gpu_gib={:.4f}".format(n_points, bytes_to_gib(peak)))
        except Exception:
            peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() and device.type == "cuda" else 0
            print("diagnostic_n={} status=error peak_gpu_gib={:.4f}".format(n_points, bytes_to_gib(peak)))
            print(traceback.format_exc())
            break
    print("=== PLANB GP DIAGNOSTIC END ===")


if RUN_DIAGNOSTIC:
    run_diagnostic()
    raise SystemExit(0)
model, likelihood, train_target = build_gp_components()
run_training_epoch(model, likelihood, training_inputs, train_target)

model.eval()
likelihood.eval()


def get_test_results(test_input_array, test_output_array):
    x_test = torch.tensor(test_input_array.reshape(-1, 8)).contiguous()
    x_test = identity_normalize(x_test, train_mean.cpu(), train_std.cpu()).to(device=device, dtype=torch.float64)
    y_test = torch.tensor(test_output_array.reshape(-1, N_TASKS)).contiguous().to(device=device, dtype=torch.float64)
    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=max(1, len(dataset)), shuffle=False)
    means = []
    lows = []
    highs = []
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = model(x_batch)
            _ = mse_loss(preds.mean, y_batch)
            lower, upper = preds.confidence_region()
            means.append(preds.mean.cpu().numpy())
            lows.append(lower.cpu().numpy())
            highs.append(upper.cpu().numpy())
    mean_arr = np.concatenate(means, axis=0).reshape(-1, N_STATIONS, N_TASKS)
    low_arr = np.concatenate(lows, axis=0).reshape(-1, N_STATIONS, N_TASKS)
    high_arr = np.concatenate(highs, axis=0).reshape(-1, N_STATIONS, N_TASKS)
    return mean_arr, low_arr, high_arr


mean_by_year = {}
low_by_year = {}
high_by_year = {}
actual_by_year = {}

for year, x_in, y_out in zip(TEST_YEAR_LABELS, test_inputs, test_outputs):
    mean_arr, low_arr, high_arr = get_test_results(x_in, y_out)
    mean_by_year[year] = mean_arr
    low_by_year[year] = low_arr
    high_by_year[year] = high_arr
    actual_by_year[year] = y_out
    joblib.dump(mean_arr, os.path.join(RESULTS_PREDICT_DIR, "{}_weekly_gp_window_predict.pkl".format(year - 1)))
    joblib.dump(low_arr, os.path.join(RESULTS_PREDICT_DIR, "{}_weekly_gp_low_window_predict.pkl".format(year - 1)))
    joblib.dump(high_arr, os.path.join(RESULTS_PREDICT_DIR, "{}_weekly_gp_high_window_predict.pkl".format(year - 1)))

joblib.dump(mean_by_year, os.path.join(PLANB_ROOT, "planB_feb1_24week_mean.pkl"))
joblib.dump(low_by_year, os.path.join(PLANB_ROOT, "planB_feb1_24week_low.pkl"))
joblib.dump(high_by_year, os.path.join(PLANB_ROOT, "planB_feb1_24week_high.pkl"))
joblib.dump(actual_by_year, os.path.join(PLANB_ROOT, "planB_feb1_24week_actual.pkl"))
