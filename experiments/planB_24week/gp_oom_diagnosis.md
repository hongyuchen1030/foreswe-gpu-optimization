# Plan B 24-Week GP OOM Diagnosis

## Bottom Line

The GP failure came from the adapted Plan B script:

- [planB_24week_gp_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py)

The crash happened on the forward pass at:

- [planB_24week_gp_model.py:141](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py:141)

It failed before loss evaluation, backward, prediction, or saving. The call stack shows the failure occurs while `gpytorch` is densifying and Cholesky-factorizing the inducing-point covariance inside the variational strategy.

The smallest change that makes the current Plan B GP feasible is:

- keep the current sparse variational multioutput GP
- reduce `PLANB_GP_INDUCING` from `7000` to something like `2048` or `1024`

This is already Option C, but it is only a parameter change, not a model rewrite.

## 1. Exact OOM Location

- Failed script: [planB_24week_gp_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py)
- Failing line: `output = model(x_batch)`
- Code location: [planB_24week_gp_model.py:141](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py:141)
- Phase: forward pass during training
- More precise phase: variational strategy Cholesky on the inducing-point covariance

Full traceback from [planB_24week_gp_interactive_80g.log](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/planB_24week_gp_interactive_80g.log):

```text
Traceback (most recent call last):
  File "/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py", line 141, in <module>
    output = model(x_batch)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/models/approximate_gp.py", line 114, in __call__
    return self.variational_strategy(inputs, prior=prior, **kwargs)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/variational/lmc_variational_strategy.py", line 197, in __call__
    latent_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/variational/variational_strategy.py", line 272, in __call__
    return super().__call__(x, prior=prior, **kwargs)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/variational/_variational_strategy.py", line 347, in __call__
    return super().__call__(
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/module.py", line 31, in __call__
    outputs = self.forward(*inputs, **kwargs)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/variational/variational_strategy.py", line 203, in forward
    L = self._cholesky_factor(induc_induc_covar)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/utils/memoize.py", line 76, in g
    return _add_to_cache_ignore_args(self, cache_name, method(self, *args, **kwargs))
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/variational/variational_strategy.py", line 104, in _cholesky_factor
    L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/operators/_linear_operator.py", line 2993, in to_dense
    return obj.to_dense()
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/utils/memoize.py", line 59, in g
    return _add_to_cache(self, cache_name, method(self, *args, **kwargs), *args, kwargs_pkl=kwargs_pkl)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/operators/sum_linear_operator.py", line 81, in to_dense
    return (sum(linear_op.to_dense() for linear_op in self.linear_ops)).contiguous()
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/operators/sum_linear_operator.py", line 81, in <genexpr>
    return (sum(linear_op.to_dense() for linear_op in self.linear_ops)).contiguous()
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/utils/memoize.py", line 59, in g
    return _add_to_cache(self, cache_name, method(self, *args, **kwargs), *args, kwargs_pkl=kwargs_pkl)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/gpytorch/lazy/lazy_evaluated_kernel_tensor.py", line 410, in to_dense
    return self.evaluate_kernel().to_dense()
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/utils/memoize.py", line 59, in g
    return _add_to_cache(self, cache_name, method(self, *args, **kwargs), *args, kwargs_pkl=kwargs_pkl)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/operators/sum_linear_operator.py", line 81, in to_dense
    return (sum(linear_op.to_dense() for linear_op in self.linear_ops)).contiguous()
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/operators/sum_linear_operator.py", line 81, in <genexpr>
    return (sum(linear_op.to_dense() for linear_op in self.linear_ops)).contiguous()
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/utils/memoize.py", line 59, in g
    return _add_to_cache(self, cache_name, method(self, *args, **kwargs), *args, kwargs_pkl=kwargs_pkl)
  File "/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/lib/python3.8/site-packages/linear_operator/operators/constant_mul_linear_operator.py", line 176, in to_dense
    return res * self.expanded_constant
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.83 GiB. GPU 0 has a total capacity of 39.49 GiB of which 1011.81 MiB is free. Process 1337420 has 760.00 MiB memory in use. Process 1857479 has 1.04 GiB memory in use. Process 9648 has 16.13 GiB memory in use. Including non-PyTorch memory, this process has 20.56 GiB memory in use. Of the allocated memory 19.82 GiB is allocated by PyTorch, and 224.66 MiB is reserved by PyTorch but unallocated.
```

Note:

- the launcher requested an `hbm80g` GPU
- the traceback shows the process actually saw a `39.49 GiB` GPU

## 2. Exact Tensor Shapes Before GP Training

Training tensors loaded from attention outputs:

- `training_inputs_list_len = 3297`
- `training_outputs_list_len = 3297`
- `training_inputs_item0_shape = (1, 512, 8)`
- `training_outputs_item0_shape = (1, 512, 24)`

After flattening in the GP script:

- `train_x shape = (1688064, 8)`
- `train_y shape = (1688064, 24)`
- raw saved dtype before GP casting: `float32`
- GP training dtype after `.to(..., dtype=torch.float64)`: `float64`
- device in the real GPU diagnostic: `cuda:0`
- number of training points `N = 1,688,064`
- output dimension `D = 24`

Test tensors used by the GP:

- `test1_x shape = (512, 8)`
- `test1_y shape = (512, 24)`
- `test2_x shape = (512, 8)`
- `test2_y shape = (512, 24)`
- `test3_x shape = (512, 8)`
- `test3_y shape = (512, 24)`
- `test4_x shape = (512, 8)`
- `test4_y shape = (512, 24)`
- `test5_x shape = (512, 8)`
- `test5_y shape = (512, 24)`

These test shapes come from the fixed Feb 1 issue date for each held-out year, one issue date times `512` stations.

## 3. Exact GP Model Type

From [planB_24week_gp_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py):

- GP base class: `ApproximateGP`
- not `ExactGP`
- multioutput model, not 24 independent scalar models
- variational strategy: `LMCVariationalStrategy(VariationalStrategy(...))`
- inducing points: yes
- default inducing count: `PLANB_GP_INDUCING = 7000`
- number of latent GPs: `5`
- number of tasks: `24`
- kernel: `ScaleKernel(RBFKernel + LinearKernel)`
- likelihood: `MultitaskGaussianLikelihood(num_tasks=24)`
- forward return: `gpytorch.distributions.MultivariateNormal`
- because it uses `LMCVariationalStrategy`, the effective model is multitask rather than independent outputs
- the traceback confirms that `to_dense(induc_induc_covar)` is executed, so the inducing covariance is explicitly materialized as a dense matrix

The original weekly GP file is structurally the same pattern, but with:

- `num_tasks = 4`
- `training_outputs.reshape(-1, 4)`

## 4. Memory Estimate

Using the actual Plan B training size `N = 1,688,064` and `D = 24`:

Float32 dense covariance estimates:

- `N x N`: `10615.4385 GiB`
- `(N*D) x (N*D)`: `6114492.5625 GiB`

Float64 dense covariance estimates, which matter because the current GP code calls `torch.set_default_dtype(torch.float64)` and casts tensors to `float64`:

- `N x N`: `21230.8770 GiB`
- `(N*D) x (N*D)`: `12228985.1250 GiB`

These are far beyond 80 GB.

However, that is not what failed first. The actual failure happened earlier in the sparse variational machinery while building the dense inducing covariance:

- latent count `L = 5`
- inducing points per latent `M = 7000`
- float64 dense inducing covariance per latent: `0.3651 GiB`
- all 5 latents together: `1.8254 GiB`

That `1.8254 GiB` matches the traceback almost exactly:

- OOM attempted allocation: `1.83 GiB`

So the immediate OOM is caused by the chosen inducing size, not by constructing the full `N x N` covariance.

## 5. Which Issue Dates Are Used for GP Training

The GP does **not** train on Feb 1 only.

Evidence:

- [planB_24week_att_model.py:113-145](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_att_model.py:113) builds `training_data_indices` from `dataloader.getTrainingDataIndices()`, then trims only the final `forecasting_window * forecasting_step_size` tail
- [planB_24week_att_model.py:393-396](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_att_model.py:393) collects representations for **all** `training_data_indices` and saves them as GP training tensors
- [planB_shape_check.txt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_shape_check.txt) records `train_indices_count=3297`

Therefore the current GP uses:

- all training issue dates returned by the ForeSWE dataloader
- across the non-test training years
- not just Feb 1

After station flattening:

- `3297` issue dates x `512` stations = `1,688,064` GP training rows

## 6. Safe Diagnostic Mode Implemented

Added to [planB_24week_gp_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py):

- `MAX_GP_TRAIN_POINTS`
- `GP_OUTPUT_MODE = "single_lead" | "multioutput"`
- `DEVICE = "cpu" | "cuda"`
- `PLANB_GP_DIAGNOSTIC`
- `PLANB_GP_DIAGNOSTIC_SIZES`

This diagnostic mode:

- prints shapes, dtype, device, `N`, `D`
- prints GP class/likelihood/strategy/kernel info
- computes covariance-size estimates
- runs small-N training smoke tests
- records peak GPU memory via `torch.cuda.max_memory_allocated()`

### GPU Diagnostic Results

Configuration:

- `PLANB_GP_EPOCHS=1`
- `PLANB_GP_BATCH=512`
- `PLANB_GP_INDUCING=128`

Current multioutput model (`GP_OUTPUT_MODE=multioutput`):

- `N=1000`: peak GPU memory `0.4572 GiB`
- `N=5000`: peak GPU memory `0.4614 GiB`
- `N=10000`: peak GPU memory `0.4651 GiB`

Single-lead scalar GP diagnostic (`GP_OUTPUT_MODE=single_lead`, lead 0 only):

- `N=10000`: peak GPU memory `0.4537 GiB`

Interpretation:

- for small `N`, memory barely changes between `1000`, `5000`, and `10000`
- that is consistent with the current sparse variational setup, where the immediate memory bottleneck is dominated much more by inducing-point structures than by the total dataset size
- the current multioutput and single-lead diagnostics are also very close at `M=128`
- the real OOM is explained by `M=7000`, not by `N=10000`

## 7. Evaluation of Proposed Fixes

### A. Feb-1-only GP training

Pros:

- reduces GP rows from `1,688,064` to roughly `20 x 512 = 10,240`
- aligns training issue dates with the fixed-issue Plan B evaluation setup

Cons:

- requires a data-selection code change
- does not address the immediate `M=7000` inducing OOM by itself

Verdict:

- useful if you want the GP to match the Feb 1-only Plan B framing
- not the smallest fix for the current crash

### B. Independent GP per lead

Pros:

- simplifies each GP to scalar output
- avoids LMC multitask coupling

Cons:

- requires looping over 24 leads and changing prediction/output handling
- more code churn than needed for the current OOM
- at small `N`, peak GPU memory was only slightly lower than the current multioutput diagnostic when both used `M=128`

Verdict:

- feasible, but not the least-change fix

### C. Sparse variational GP with smaller inducing count

Pros:

- already the current model family
- only needs changing `PLANB_GP_INDUCING`
- directly targets the actual failure site
- the traceback allocation (`1.83 GiB`) matches the dense float64 inducing covariance for `L=5, M=7000`

Cons:

- may affect fit quality versus a very large inducing set
- does not reduce wall-clock time due to the huge number of minibatches as much as Feb 1-only training would

Verdict:

- **least code change**
- **best first fix**

### D. CPU GP

Pros:

- avoids GPU capacity contention

Cons:

- much slower
- does not fix the underlying oversized inducing configuration
- if the code ever moves toward denser covariance use, CPU memory pressure can still become severe

Verdict:

- fallback option, not first choice

### E. No GP, empirical intervals

Pros:

- simplest operationally
- avoids GP cost entirely

Cons:

- changes the statistical method more substantially than the other options

Verdict:

- pragmatic fallback if GP uncertainty is not essential
- not the least-change path if you still want a GP head

## Recommended Fix

Recommended first step:

- keep the current multioutput sparse variational GP
- reduce `PLANB_GP_INDUCING` from `7000` to `2048`

Why `2048`:

- this is still much larger than the `128` used in the smoke diagnostics
- the dense float64 inducing covariance over 5 latents is only about `0.1563 GiB`
- it directly removes the specific `1.83 GiB` dense inducing allocation that caused the crash

If you want an even more conservative starting point:

- use `PLANB_GP_INDUCING=1024`

If you want the GP training set to better match the Plan B fixed-issue use case after that:

- then implement Option A as a second refinement

## 8. Exact Next Feasible Run

Least-change next run:

```bash
PLANB_GP_SMOKE=0 \
PLANB_GP_EPOCHS=2 \
PLANB_GP_BATCH=4096 \
PLANB_GP_INDUCING=2048 \
DEVICE=cuda \
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py
```

More conservative memory-first variant:

```bash
PLANB_GP_SMOKE=0 \
PLANB_GP_EPOCHS=2 \
PLANB_GP_BATCH=4096 \
PLANB_GP_INDUCING=1024 \
DEVICE=cuda \
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py
```

Diagnostic command used for the GPU measurements:

```bash
PLANB_GP_DIAGNOSTIC=1 \
PLANB_GP_EPOCHS=1 \
PLANB_GP_BATCH=512 \
PLANB_GP_INDUCING=128 \
DEVICE=cuda \
PLANB_GP_DIAGNOSTIC_SIZES=1000,5000,10000 \
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py
```
