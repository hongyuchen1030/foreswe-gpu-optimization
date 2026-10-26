import math
import os
import random
import sys
import time
from contextlib import redirect_stdout

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PLANB_ROOT = os.getenv(
    "FSW_PLANB_ROOT",
    "/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week",
)
ORIG_REPO_ROOT = os.getenv(
    "FSW_ORIG_REPO",
    "/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting",
)
FORE_SWE_DIR = os.path.join(ORIG_REPO_ROOT, "fore_swe")
DATA_DIR = os.getenv("FSW_DATA_DIR", os.path.join(ORIG_REPO_ROOT, "data"))
HF_CACHE_ROOT = os.getenv("FSW_HF_CACHE", "/pscratch/sd/h/hyvchen/foreswe_model_cache")
OUTPUTS_DIR = os.path.join(PLANB_ROOT, "outputs")
RESULTS_ACTUAL_DIR = os.path.join(PLANB_ROOT, "results", "actual")
RESULTS_PREDICT_DIR = os.path.join(PLANB_ROOT, "results", "predict")
CHECKPOINTS_DIR = os.path.join(PLANB_ROOT, "checkpoints")
SHAPE_CHECK_PATH = os.path.join(PLANB_ROOT, "planB_shape_check.txt")

for path in [PLANB_ROOT, OUTPUTS_DIR, RESULTS_ACTUAL_DIR, RESULTS_PREDICT_DIR, CHECKPOINTS_DIR]:
    os.makedirs(path, exist_ok=True)

os.chdir(ORIG_REPO_ROOT)
sys.path.insert(0, FORE_SWE_DIR)
os.environ.setdefault("HF_HOME", os.path.join(HF_CACHE_ROOT, "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(HF_CACHE_ROOT, "sentence_transformers"))

import extended_dl_forecasting as DL  # noqa: E402
from custom_encoder_model import EncoderLayer as CustomEncoderLayer  # noqa: E402


SEED = int(os.getenv("PLANB_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU is running.")
else:
    print("CPU is running.")

forecasting_window = 24
forecasting_step_size = 7
history_daily_window = 30
history_weekly_window = 30
history_yearly_window = 5
history_weekly_step = 7
history_yearly_step = 365

PLANB_SMOKE = os.getenv("PLANB_SMOKE", "0") != "0"
PLANB_MAX_TRAIN_INDICES = int(os.getenv("PLANB_MAX_TRAIN_INDICES", "4" if PLANB_SMOKE else "0"))
PLANB_MAX_EPOCHS = int(os.getenv("PLANB_MAX_EPOCHS", "1" if PLANB_SMOKE else "6"))
ALLOW_ZERO_PROMPTS = os.getenv("PLANB_ALLOW_ZERO_PROMPTS", "0") != "0"

TRAIN_LOG_PATH = os.path.join(OUTPUTS_DIR, "planB_24week_train_log.txt")
TEST_LOG_PATH = os.path.join(OUTPUTS_DIR, "planB_24week_test_log.txt")
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "planB_24week_att_model.pt")

TEST_YEAR_LABELS = [2015, 2016, 2017, 2018, 2019]
FEB1_ISSUE_INDICES = {
    2015: 8511,
    2016: 8876,
    2017: 9241,
    2018: 9606,
    2019: 9971,
}

full_metadata = pd.read_csv(os.path.join(DATA_DIR, "snotel_location_info_2.csv"))
N_STATIONS = len(full_metadata)

print("Plan B root", PLANB_ROOT)
print("Original repo root", ORIG_REPO_ROOT)
print("Data directory", DATA_DIR)
print("Forecasting window", forecasting_window)
print("Forecasting step size", forecasting_step_size)
print("Weekly lead days", [7 * value for value in range(1, forecasting_window + 1)])
print("N_STATIONS", N_STATIONS)
print("PLANB_SMOKE", PLANB_SMOKE)
print("PLANB_MAX_TRAIN_INDICES", PLANB_MAX_TRAIN_INDICES)
print("PLANB_MAX_EPOCHS", PLANB_MAX_EPOCHS)
print("HF cache root", HF_CACHE_ROOT)

test_indices = [43, 66, 338, 463, 115, 105, 83, 306, 481, 21, 185, 136, 110, 406, 408, 132,
                120, 197, 7, 301, 375, 236, 269, 287, 49, 241, 212, 145, 264, 435, 73, 277, 364,
                438, 372, 368, 243, 493, 497, 14, 32, 433, 313, 213, 42, 475, 160, 190, 281, 285,
                316, 461, 154, 424, 310, 340, 179, 167, 193, 466, 56, 158, 365, 492, 432, 415, 68,
                409, 76, 148, 267, 227, 91, 189, 472, 162, 147, 395, 113, 2, 271, 357, 347, 335, 507,
                465, 57, 19, 442, 381, 398, 79, 337, 501, 41, 339, 331, 63, 359, 232, 323, 258]
test_indices = sorted(test_indices)

dataloader = DL.DataLoader(test_indices=test_indices, temp_window=7)
all_locations_distances, all_locations_angularities = dataloader.getDistanceAndAngularity()
dataloader.cleanDataJunk()
dataloader.normalizeData()
dataloader.addDayInfo()
dataloader.prepareInputsAndOutputs()
model_inputs, model_outputs = dataloader.getModelInputsAndOutputs()
dataloader.completeInputsAndOutputs()
training_data_indices = np.array(dataloader.getTrainingDataIndices())
training_data_indices = training_data_indices[: -(forecasting_window * forecasting_step_size)]
testing_idx_arrays = dataloader.getTestingDataIndices()
train_loc_indices = dataloader.getTrainingIndices()
try:
    train_prompts, test_prompts, all_prompts = dataloader.getModelPrompts()
    PROMPT_MODE = "sentence-transformer"
except Exception as exc:
    if not ALLOW_ZERO_PROMPTS:
        raise RuntimeError(
            "Prompt loading failed and PLANB_ALLOW_ZERO_PROMPTS is disabled. "
            "Set FSW_HF_CACHE correctly or explicitly allow zero prompts for debugging only. "
            "Original error: {}".format(exc)
        )
    print("Prompt loading failed; falling back to zero prompts. Error: {}".format(exc))
    prompt_dim = 1536
    all_prompts = np.zeros((1, N_STATIONS, prompt_dim), dtype=np.float32)
    train_prompts = all_prompts[:, train_loc_indices, :]
    test_prompts = all_prompts[:, test_indices, :]
    PROMPT_MODE = "zero-fallback"
max_swe, min_swe = dataloader.getMaxMinSWE()

all_dates = np.array(dataloader.all_dates)
test_issue_indices = {}
for year, indices in zip(TEST_YEAR_LABELS, testing_idx_arrays):
    matched = [idx for idx in indices if str(all_dates[idx])[:10] == "{}-02-01".format(year)]
    if len(matched) != 1:
        raise RuntimeError("Expected exactly one Feb 1 issue index for {}, found {}".format(year, matched))
    test_issue_indices[year] = np.array(matched, dtype=int)

if PLANB_MAX_TRAIN_INDICES > 0:
    training_data_indices = training_data_indices[:PLANB_MAX_TRAIN_INDICES]

train_prompts = torch.from_numpy(train_prompts).to(device=device, dtype=torch.float32)
test_prompts = torch.from_numpy(test_prompts).to(device=device, dtype=torch.float32)
all_prompts = torch.from_numpy(all_prompts).to(device=device, dtype=torch.float32)
model_inputs = torch.from_numpy(model_inputs).to(device=device, dtype=torch.float32)
model_outputs = torch.from_numpy(model_outputs).to(device=device, dtype=torch.float32)


def write_shape_check():
    lead_days = [7 * value for value in range(1, forecasting_window + 1)]
    lines = [
        "forecasting_window={}".format(forecasting_window),
        "forecasting_step_size={}".format(forecasting_step_size),
        "target_formula=index + value * 7 for value in range(1, 25)",
        "history_daily_window={}".format(history_daily_window),
        "history_weekly_window={} step={}".format(history_weekly_window, history_weekly_step),
        "history_yearly_window={} step={}".format(history_yearly_window, history_yearly_step),
        "lead_days={}".format(lead_days),
        "model_inputs_shape={}".format(tuple(model_inputs.shape)),
        "model_outputs_shape={}".format(tuple(model_outputs.shape)),
        "station_count={}".format(N_STATIONS),
        "train_indices_count={}".format(len(training_data_indices)),
        "planb_smoke={}".format(PLANB_SMOKE),
        "planb_max_train_indices={}".format(PLANB_MAX_TRAIN_INDICES),
        "planb_max_epochs={}".format(PLANB_MAX_EPOCHS),
        "prompt_mode={}".format(PROMPT_MODE),
    ]
    for year in TEST_YEAR_LABELS:
        idx = int(test_issue_indices[year][0])
        lines.append("feb1_issue_index_{}={} date={}".format(year, idx, all_dates[idx]))
    with open(SHAPE_CHECK_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


write_shape_check()


class CrossVarTransformer(nn.Module):
    def __init__(self, encoder_input_dim=13, batch_size=7, sp_tp_dim=6, model_dim=512, prompt_dim=1536, enc_layers_count=8, nhead=16):
        super().__init__()
        self.model_dim = model_dim
        self.prompt_dim = prompt_dim
        self.encoder_input_dim = encoder_input_dim
        self.batch_size = batch_size
        self.sp_tp_dim = sp_tp_dim
        self.nhead = nhead
        self.enc_layers_count = enc_layers_count

        self.variable_embeddings = nn.Parameter(torch.zeros(1, encoder_input_dim, self.model_dim), requires_grad=True)
        variable_embeddings = self.getVariableEmbeddingInitialized(self.variable_embeddings.shape[-1], np.arange(self.encoder_input_dim))
        self.variable_embeddings.data.copy_(variable_embeddings.unsqueeze(0))

        self.token_embeds = nn.ModuleList([torch.nn.Linear(self.batch_size, self.model_dim) for _ in range(self.encoder_input_dim)])

        self.sp_tp_prompt_embed = nn.Sequential(
            torch.nn.Linear(self.sp_tp_dim, int(self.model_dim / 2)),
            nn.GELU(),
            nn.Dropout(p=0.05),
            torch.nn.Linear(int(self.model_dim / 2), int(self.model_dim)),
        )

        self.prompt_embedding = nn.Linear(self.prompt_dim, self.model_dim)
        self.variable_attn = nn.MultiheadAttention(self.model_dim, self.nhead, batch_first=True)
        self.encoder_layers = nn.ModuleList([CustomEncoderLayer(d_model=self.model_dim, num_heads=self.nhead) for _ in range(self.enc_layers_count)])

    def getVariableEmbeddingInitialized(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return torch.from_numpy(emb).to(device=device, dtype=torch.float32)

    def forward(self, encoder_inputs, prompt_inputs, distance_pairs, angularity_pairs):
        inp_batch_size, sequence_length, input_dim = encoder_inputs.shape
        encoder_inputs_historical = encoder_inputs[:, :, 6:]
        encoder_inputs_var_temp = encoder_inputs_historical.permute(1, 0, 2).reshape(sequence_length, inp_batch_size, -1).permute(0, 2, 1)
        inputs_sp_tp = encoder_inputs[-1, :, :6].reshape(sequence_length, -1)
        sp_tp_embedding = self.sp_tp_prompt_embed(inputs_sp_tp).reshape(1, sequence_length, int(self.model_dim)).permute(1, 0, 2)
        prompt_embedding_x = self.prompt_embedding(prompt_inputs.reshape(-1, self.prompt_dim)).reshape(1, sequence_length, self.model_dim).permute(1, 0, 2)

        embeds = []
        for index in range(self.encoder_input_dim):
            embeds.append(self.token_embeds[index](encoder_inputs_var_temp[:, index, :].reshape(-1, self.batch_size)))
        embedded_var_input_x = torch.stack(embeds, dim=0).to(device=device, dtype=torch.float32).permute(1, 0, 2)
        variable_embedding = self.variable_embeddings.repeat(sequence_length, 1, 1)
        embedded_var_input_x = embedded_var_input_x + variable_embedding
        query_embedding = sp_tp_embedding + prompt_embedding_x
        var_encoded_input_x, _ = self.variable_attn(query_embedding, embedded_var_input_x, embedded_var_input_x)
        var_encoded_input_x = var_encoded_input_x.squeeze(1).reshape(1, sequence_length, self.model_dim)
        final_output = var_encoded_input_x + query_embedding.permute(1, 0, 2)
        return final_output.reshape(1, sequence_length, self.model_dim)


class SWETransformer(nn.Module):
    def __init__(self, model_dim=1024, prompt_dim=1536, n_output_heads=1, enc_layers_count=8, nhead=16):
        super().__init__()
        self.model_dim = model_dim
        self.prompt_dim = prompt_dim
        self.n_output_heads = n_output_heads
        self.nhead = nhead
        self.enc_layers_count = enc_layers_count
        self.historical_daily_att = CrossVarTransformer(batch_size=30, model_dim=self.model_dim, prompt_dim=self.prompt_dim)
        self.historical_monthly_att = CrossVarTransformer(batch_size=30, model_dim=self.model_dim, prompt_dim=self.prompt_dim)
        self.historical_yearly_att = CrossVarTransformer(batch_size=5, model_dim=self.model_dim, prompt_dim=self.prompt_dim)
        self.encoder_layers = nn.ModuleList([CustomEncoderLayer(d_model=3 * self.model_dim, num_heads=self.nhead) for _ in range(self.enc_layers_count)])
        self.output_layers = nn.Sequential(
            torch.nn.Linear(self.model_dim * 3, self.model_dim),
            nn.GELU(),
            nn.Dropout(p=0.05),
            torch.nn.Linear(self.model_dim, 128),
            nn.GELU(),
            torch.nn.Linear(128, 8),
            nn.GELU(),
        )
        self.final_output = torch.nn.Linear(8, self.n_output_heads)

    def forward(self, historical_daily_inputs, historical_weekly_inputs, historical_yearly_inputs, loc_prompt_inputs, token_distances, token_angularities):
        historical_daily_encoded = self.historical_daily_att(historical_daily_inputs, loc_prompt_inputs, token_distances, token_angularities)
        historical_weekly_encoded = self.historical_monthly_att(historical_weekly_inputs, loc_prompt_inputs, token_distances, token_angularities)
        historical_yearly_encoded = self.historical_yearly_att(historical_yearly_inputs, loc_prompt_inputs, token_distances, token_angularities)
        merged_input_x = torch.cat((historical_daily_encoded, historical_weekly_encoded, historical_yearly_encoded), dim=2)
        final_output = merged_input_x
        for encoder_layer in self.encoder_layers:
            final_output = encoder_layer(final_output, token_distances, token_angularities)
        final_output = final_output + merged_input_x
        final_representation = self.output_layers(final_output.reshape(-1, 3 * self.model_dim))
        outputs = self.final_output(final_representation).reshape(1, historical_daily_inputs.shape[1], self.n_output_heads)
        final_representation = final_representation.reshape(1, historical_daily_inputs.shape[1], 8)
        return outputs, final_representation


swe_model = SWETransformer(model_dim=1024, prompt_dim=1536, n_output_heads=forecasting_window, nhead=16).to(device=device)
mae_error = nn.MSELoss()
optimizer_sp = torch.optim.AdamW(swe_model.parameters(), lr=0.000005, weight_decay=0.0001)
scheduler_sp = torch.optim.lr_scheduler.StepLR(optimizer_sp, step_size=4, gamma=0.65, last_epoch=-1, verbose=False)


def get_inputs(index, loc_indices, window, step_size):
    indices = sorted([index - (value * step_size) for value in range(0, window)])
    return model_inputs[:, loc_indices, :][indices, :, :].reshape(window, len(loc_indices), -1)


def get_outputs(index, loc_indices, window, step_size):
    indices = sorted([index + (value * step_size) for value in range(1, window + 1)])
    return model_outputs[:, loc_indices, :][indices, :, :].permute(1, 0, 2).reshape(len(loc_indices), -1)


def get_example(index):
    all_indices = sorted(list(train_loc_indices) + list(test_indices))
    daily = get_inputs(index, all_indices, history_daily_window, 1)
    weekly = get_inputs(index, all_indices, history_weekly_window, history_weekly_step)
    yearly = get_inputs(index, all_indices, history_yearly_window, history_yearly_step)
    outputs = get_outputs(index, all_indices, forecasting_window, forecasting_step_size)
    distances = torch.from_numpy(all_locations_distances[all_indices, :][:, all_indices]).to(device=device, dtype=torch.float32)
    angularities = torch.from_numpy(all_locations_angularities[all_indices, :][:, all_indices]).to(device=device, dtype=torch.float32)
    return (
        torch.tensor(daily).to(device=device, dtype=torch.float32),
        torch.tensor(weekly).to(device=device, dtype=torch.float32),
        torch.tensor(yearly).to(device=device, dtype=torch.float32),
        torch.tensor(outputs).to(device=device, dtype=torch.float32),
        distances,
        angularities,
        all_indices,
    )


def train_epoch(train_indices, epoch_number, out_stream):
    total_loss = 0.0
    for index, input_index in enumerate(train_indices):
        daily, weekly, yearly, target, distances, angularities, all_indices = get_example(int(input_index))
        output, _ = swe_model(daily, weekly, yearly, all_prompts, distances, angularities)
        loss = mae_error(output.reshape(daily.shape[1], -1), target)
        total_loss += loss.item()
        optimizer_sp.zero_grad()
        loss.backward()
        optimizer_sp.step()
        print(
            "Epoch {} batch {} loss {} output_shape {}".format(
                epoch_number, index + 1, loss.item(), tuple(output.shape)
            ),
            flush=True,
            file=out_stream,
        )
    avg_loss = total_loss / len(train_indices)
    print("Epoch {} avg_loss {}".format(epoch_number, avg_loss), flush=True, file=out_stream)
    return avg_loss


with open(TRAIN_LOG_PATH, "w", buffering=1) as train_log:
    print("Plan B training log path: {}".format(TRAIN_LOG_PATH), file=train_log, flush=True)
    print("Checkpoint path: {}".format(CHECKPOINT_PATH), file=train_log, flush=True)
    print("Output horizons days: {}".format([7 * value for value in range(1, forecasting_window + 1)]), file=train_log, flush=True)
    epoch_number = 1
    best_loss = float("inf")
    if os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        swe_model.load_state_dict(checkpoint["swe_model_dict"])
        optimizer_sp.load_state_dict(checkpoint["optimizer_dict"])
        epoch_number = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        print("Resuming from epoch {}".format(epoch_number), file=train_log, flush=True)
    start_time = time.time()
    while epoch_number <= PLANB_MAX_EPOCHS:
        swe_model.train()
        epoch_loss = train_epoch(training_data_indices, epoch_number, train_log)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch_number + 1,
                    "best_loss": best_loss,
                    "swe_model_dict": swe_model.state_dict(),
                    "optimizer_dict": optimizer_sp.state_dict(),
                },
                CHECKPOINT_PATH,
            )
        epoch_number += 1
        scheduler_sp.step()
    print("Elapsed_seconds={}".format(time.time() - start_time), file=train_log, flush=True)


if os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    swe_model.load_state_dict(checkpoint["swe_model_dict"])
    optimizer_sp.load_state_dict(checkpoint["optimizer_dict"])

swe_model.eval()


def collect_representations(indices):
    input_learnt_representations = []
    predicted_outputs = []
    actual_outputs = []
    with torch.no_grad():
        for input_index in indices:
            daily, weekly, yearly, target, distances, angularities, _ = get_example(int(input_index))
            output, final_representation = swe_model(daily, weekly, yearly, all_prompts, distances, angularities)
            input_learnt_representations.append(final_representation.detach().cpu())
            predicted_outputs.append(output.detach().cpu().reshape(1, -1, forecasting_window))
            actual_outputs.append(target.detach().cpu().reshape(1, -1, forecasting_window))
    return input_learnt_representations, predicted_outputs, actual_outputs


train_reps, train_preds, train_actuals = collect_representations(training_data_indices)
torch.save(train_reps, os.path.join(OUTPUTS_DIR, "forecasting_modified_weekly_att_inputs_for_gp_act.pt"))
torch.save(train_preds, os.path.join(OUTPUTS_DIR, "forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt"))
torch.save(train_actuals, os.path.join(OUTPUTS_DIR, "forecasting_modified_weekly_att_outputs_for_gp_act.pt"))

feb1_attention_actual = {}
feb1_attention_predicted = {}

with open(TEST_LOG_PATH, "w", buffering=1) as test_log:
    print("Plan B test log path: {}".format(TEST_LOG_PATH), file=test_log, flush=True)
    for year_index, year in enumerate(TEST_YEAR_LABELS, start=1):
        issue_indices = test_issue_indices[year]
        reps, preds, actuals = collect_representations(issue_indices)
        pred_arr = preds[0].numpy().reshape(1, N_STATIONS, forecasting_window)
        actual_arr = actuals[0].numpy().reshape(1, N_STATIONS, forecasting_window)
        feb1_attention_actual[year] = actual_arr
        feb1_attention_predicted[year] = pred_arr
        torch.save(reps[0].numpy().reshape(1, N_STATIONS, 8), os.path.join(OUTPUTS_DIR, "weekly_test{}_inputs_for_gp_act.pt".format(year_index)))
        joblib.dump(actual_arr, os.path.join(RESULTS_ACTUAL_DIR, "{}_weekly_window_actual.pkl".format(year - 1)))
        joblib.dump(pred_arr, os.path.join(RESULTS_PREDICT_DIR, "{}_weekly_window_predict.pkl".format(year - 1)))
        print(
            "Year {} issue_date {} issue_index {} actual_shape {} pred_shape {}".format(
                year, all_dates[int(issue_indices[0])], int(issue_indices[0]), actual_arr.shape, pred_arr.shape
            ),
            file=test_log,
            flush=True,
        )

joblib.dump(feb1_attention_actual, os.path.join(PLANB_ROOT, "planB_feb1_attention_actual_by_year.pkl"))
joblib.dump(feb1_attention_predicted, os.path.join(PLANB_ROOT, "planB_feb1_attention_predicted_by_year.pkl"))
