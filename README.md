# GPU Monte Carlo risk engine

A book of equities and European options is priced with hand-written CUDA kernels and revalued under correlated geometric Brownian motion. Value-at-Risk and expected shortfall come from a GPU histogram of the P&L. The same `mc_engine` binary runs on this PC and inside the AWS Batch image.

## What ran on this machine

NVIDIA GeForce RTX 3050 Laptop GPU, compute capability 8.6, 16 SMs. The driver is CUDA 13.1. There is no `nvcc` on the machine, so the engine compiles `kernels/mc_risk.cu` with NVRTC (the CUDA 12.9 runtime compiler, installed under `D:\HPC_L2\.cuda_pkgs`).

Self-test (`build\mc_engine.exe selftest`):

- Philox4x32-10 matches the published test vectors on the CPU and on the GPU.
- Black-Scholes ATM call (S=K=100, r=5%, vol=20%, T=1) is 10.450584, and the device formula matches the host.
- One-day 99% VaR and expected shortfall of a unit equity match the lognormal formula.
- A 2 million path call lands inside sampling error of Black-Scholes.
- A geometric Asian matches its closed form. The arithmetic price uses that geometric average as a control variate.
- European pricing is about 9.7× faster than 12 CPU threads at 1 million paths.

Sample four-name book, 100 million scenarios, 1-day horizon:

| | |
| --- | --- |
| Portfolio value | 50,699.92 |
| 95% VaR | 853.34 |
| 99% VaR | 1,211.44 |
| 99% expected shortfall | 1,390.12 |
| GPU job time | 8.39 s |
| Pricing throughput | 278 million paths/s (8.5× vs 12 CPU threads) |
| Scenario throughput | 25.6 million paths/s |
| Device value vs analytic value | same to the printed digits |

Option Monte Carlo prices sit inside the 95% confidence interval of the Black-Scholes marks. The AAPL geometric Asian at 100 million paths is 13.39356 against a closed form of 13.39300.

Spots in the sample book are illustrative, not live prices.

## Run locally

```powershell
powershell -File D:\HPC_L2\build.ps1
powershell -File D:\HPC_L2\run_dashboard.ps1
```

Open http://127.0.0.1:8765 . Presets are 1 million, 10 million, and 100 million scenarios.

`build.ps1` sets `TEMP`, `TMP`, and the pip cache to `D:\HPC_L2` before it downloads NVRTC or compiles. The engine only links `KERNEL32.dll` and `msvcrt.dll`.

Direct engine call:

```powershell
D:\HPC_L2\build\mc_engine.exe run D:\HPC_L2\tmp\bench100.txt D:\HPC_L2\build\bench100.json
```

## Model

- Price paths are geometric Brownian motion. European prices use antithetic normals. The discount is the continuous risk-free rate.
- Portfolio scenarios apply a Cholesky factor of the correlation matrix, step every underlying to the horizon, and revalue. Equities are marked at the simulated spot. Options are Black-Scholes with the remaining maturity.
- Drift is the price drift. The sample uses rate minus dividend (risk-neutral). One day is `1/252` of a year.
- VaR is the loss at the 95% and 99% P&L quantiles. Expected shortfall is the average P&L inside that tail, read back from the histogram. A negative VaR would mean that tail is still a gain.
- The random number generator is Philox4x32-10. Nothing is drawn from cuRAND.

## AWS

The container image is `docker/Dockerfile` (`nvidia/cuda:12.4.1-devel`). It compiles this same `src/engine.c` and runs `kernels/mc_risk.cu` through the image's libnvrtc. `docker/entrypoint.sh` copies the job from S3, runs the engine, and writes the JSON back.

`aws/template.yaml` is a SAM stack:

1. HTTP API `POST /jobs` and `GET /jobs/{job_id}`
2. Lambda validates the book, writes `job.txt` to S3, and submits Batch
3. Spot compute environment on `g4dn.xlarge` (T4) and `g5.xlarge` (A10G), scaled to zero when idle
4. The container role can read and write only the `jobs/` prefix

Packaging the Lambda zip:

```powershell
powershell -File D:\HPC_L2\build.ps1
```

That refreshes `lambda_pkg\handler.py` and `lambda_pkg\jobtext.py`. Deploy with your subnet and security group:

```powershell
sam deploy --template-file D:\HPC_L2\aws\template.yaml --guided --parameter-overrides ImageUri=ACCOUNT.dkr.ecr.REGION.amazonaws.com/mc-risk:latest SubnetIds=subnet-xxx SecurityGroupIds=sg-xxx
```

Build and push the image from a host whose Docker data root is not the C: drive, or from CI. On-demand list prices used by the dashboard are us-east-1 Linux: g4dn.xlarge `$0.526/h`, g5.xlarge `$1.006/h` (October 2026). The page also scales the measured time by published FP32 peaks (T4 8.1 TFLOPS, A10G 31.2 TFLOPS, this GPU reported 6.14 TFLOPS at 1500 MHz). That scale is an estimate. Spot prices move and are usually lower than those list rates.

## Layout

| Path | Role |
| --- | --- |
| `kernels/mc_risk.cu` | Philox, European pricing, scenario VaR, Asian control variate |
| `src/engine.c` | Driver API host, CPU baseline, JSON result |
| `dashboard/` | Browser and local API |
| `common/jobtext.py` | Validation and the job text format |
| `aws/` | Lambda and SAM |
| `docker/` | Batch image |
