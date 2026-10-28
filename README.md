# HPC CUDA AWS

GPU Monte Carlo risk engine. Custom CUDA kernels price a book of equities and European options, then revalue that book under correlated geometric Brownian motion to produce Value-at-Risk and expected shortfall. The same engine binary runs on a workstation and inside an AWS Batch GPU container.

## Features

- Hand-written CUDA kernels. Pricing does not call cuRAND or a vendor pricing library.
- Philox4x32-10 counter-based RNG, checked against the published test vectors.
- European prices use antithetic variates and are checked against Black-Scholes.
- Portfolio scenarios use a Cholesky factor, then a full revaluation. VaR and expected shortfall come from a GPU histogram.
- Arithmetic Asian calls use a geometric Asian control variate.
- Multi-threaded CPU baseline and a measured GPU speedup.
- Local dashboard, plus an API Gateway, Lambda, S3, and spot Batch deployment for `g4dn` and `g5`.

## Architecture

The full write-up is in [architecture.md](architecture.md): the problem, both runtimes, the Monte Carlo method, the kernels, and the measured results.

```text
Browser
  -> local dashboard (dashboard/server.py)     -> mc_engine
  -> API Gateway -> Lambda -> S3 -> AWS Batch  -> mc_engine on a GPU instance
```

Workstation jobs and Batch jobs read the same job text and write the same JSON. The host locates `kernels/mc_risk.cu` and NVRTC from the executable directory, `NVRTC_PATH`, or `CUDA_PATH`. Nothing in the source is tied to a drive letter.

## Requirements

| Tool | Used for |
| --- | --- |
| 64-bit MinGW-w64 `gcc` | Windows host build. Set `CC` if it is not on `PATH`. |
| Python 3.9 or newer | Dashboard, job validation, and the NVRTC wheel install. |
| NVIDIA driver and GPU | Local runs. The build downloads the NVRTC redistributable into `.cuda_pkgs` when `nvcc` is not installed. |
| Docker with the NVIDIA runtime | Batch image. |
| AWS SAM CLI | Stack deploy. |

On Linux the container image supplies `gcc` and `libnvrtc`.

## Quick start

From the repository root:

```powershell
powershell -File .\build.ps1
powershell -File .\run_dashboard.ps1
```

Open http://127.0.0.1:8765. The sample book can be run at 1 million, 10 million, or 100 million scenarios.

`build.ps1` resolves the repository from its own path. Compiler output, pip cache, and temporary files stay in `build\`, `.cache\`, and `tmp\` inside that repository. Those directories are gitignored.

Run the engine directly:

```powershell
.\build\mc_engine.exe selftest
.\build\mc_engine.exe run .\path\to\job.txt .\path\to\result.json
```

On Linux, after `gcc -O3 -std=c11 -o build/mc_engine src/engine.c -pthread -ldl -lm`:

```bash
./build/mc_engine selftest
./build/mc_engine run job.txt result.json
```

## Configuration

| Variable | Meaning |
| --- | --- |
| `CC` | 64-bit `gcc` used by `build.ps1`. |
| `MC_KERNEL` | Path to `mc_risk.cu`. Default: next to the repo, or beside the binary. |
| `NVRTC_PATH` | `nvrtc64_*.dll` or `libnvrtc.so.12`. Default: `.cuda_pkgs`, then `CUDA_PATH`. |
| `CUDA_PATH` | CUDA toolkit root. Its `bin` directory is searched for NVRTC. |
| `JOB_S3`, `OUT_S3` | S3 URIs used by the Batch entrypoint. |

## Model

Prices follow geometric Brownian motion. European payoffs use antithetic normals and continuous discounting at the risk-free rate.

Each portfolio scenario applies the Cholesky factor of the correlation matrix, steps every underlying to the horizon, and revalues the book. Equities are marked at the simulated spot. Options are Black-Scholes with the remaining maturity. The sample book sets price drift to rate minus dividend. One trading day is `1/252` of a year.

VaR is the loss at the 95% and 99% P&L quantiles. Expected shortfall is the average P&L inside that tail. A negative VaR means that tail is still a gain.

Sample spots are illustrative. They are not live market data.

## AWS deployment

`docker/Dockerfile` starts from `nvidia/cuda:12.4.1-devel-ubuntu22.04`, compiles `src/engine.c`, and runs `docker/entrypoint.sh`. The entrypoint downloads the job from S3, runs the engine, and uploads the JSON.

`aws/template.yaml` creates:

1. An HTTP API with `POST /jobs` and `GET /jobs/{job_id}`.
2. A Lambda that validates the book, writes the job to S3, and submits Batch.
3. A spot compute environment for `g4dn.xlarge` (T4) and `g5.xlarge` (A10G), scaled to zero when idle.
4. A job role that can read and write only the `jobs/` prefix.

`build.ps1` copies `aws/handler.py` and `common/jobtext.py` into `lambda_pkg\`, which is the SAM code location.

```powershell
sam deploy --template-file .\aws\template.yaml --guided `
  --parameter-overrides ImageUri=<account>.dkr.ecr.<region>.amazonaws.com/mc-risk:latest `
  SubnetIds=<subnet-id> SecurityGroupIds=<sg-id>
```

Build and push the image from a machine with the NVIDIA container toolkit, or from CI.

The dashboard prices us-east-1 Linux on-demand list rates, checked in October 2026: `g4dn.xlarge` at `$0.526/hour` and `g5.xlarge` at `$1.006/hour`. A second column scales the measured runtime by published FP32 peaks (T4 8.1 TFLOPS, A10G 31.2 TFLOPS). That column is an estimate, not a run on AWS. Spot prices change through the day and are usually lower than those list rates.

## Example results

These figures are from one workstation run of the sample book. They are a reference, not a performance guarantee.

| Check | Result |
| --- | --- |
| Black-Scholes ATM call, S = K = 100, r = 5%, vol = 20%, T = 1 | 10.450584, host and device agree |
| 100 million scenarios, 1-day horizon | portfolio value 50,699.92 |
| 95% / 99% VaR | 853.34 / 1,211.44 |
| 99% expected shortfall | 1,390.12 |
| GPU job time | 8.39 s |
| European pricing | 278 million paths/s, about 8.5× a 12-thread CPU |
| Scenario revaluation | 25.6 million paths/s |

Option Monte Carlo prices fell inside the Black-Scholes 95% confidence interval. The geometric Asian matched its closed form.

## Layout

| Path | Role |
| --- | --- |
| `architecture.md` | Problem, runtimes, method, kernels, and measured results |
| `kernels/mc_risk.cu` | RNG, European pricing, scenario VaR, Asian control variate |
| `src/engine.c` | Driver API host, CPU baseline, JSON writer |
| `dashboard/` | Browser and local API |
| `common/jobtext.py` | Validation and the job text format |
| `aws/` | Lambda handler and SAM template |
| `docker/` | Batch image and entrypoint |
| `build.ps1` | Locate a 64-bit compiler, fetch NVRTC if needed, build, self-test |

## License

No license is granted by this repository. Add one before distributing the code.
