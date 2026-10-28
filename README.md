# GPU Monte Carlo Risk Engine

A local GPU Monte Carlo risk engine. Custom CUDA kernels price a synthetic book of equities and European options, then revalue that book under correlated geometric Brownian motion and return Value-at-Risk and expected shortfall. It was built and run on one Windows workstation. No model is trained.

The longer write-up is in [architecture.md](architecture.md).

## Synthetic dataset

The sample portfolio is synthetic. Spots, volatilities, dividends, rates, drifts, correlation, and positions are constants in `sample_portfolio()` in `common/jobtext.py`. Nothing was downloaded from an exchange, a data vendor, or a historical file. Nothing was fitted or trained. The labels SPX, AAPL, XOM, and GLD are names only. The numbers are not live quotes.

| Name | Spot | Volatility | Dividend |
| --- | --- | --- | --- |
| SPX | 5,200 | 16% | 1.3% |
| AAPL | 190 | 28% | 0.5% |
| XOM | 110 | 22% | 3.2% |
| GLD | 180 | 14% | 0% |

The rate on every name is 4.5%. Drift is rate minus dividend. Seed is 42. Correlation is typed in the same function. Positions are long 8 SPX, short 4 SPX 5000 puts, long 30 AAPL 200 calls, long 20 AAPL 170 puts, long 40 XOM, and long 25 GLD, plus one synthetic AAPL Asian. The dashboard form starts from this book. Edited numbers stay synthetic inputs. They are still not a market feed.

## What runs

```text
Browser -> dashboard/server.py on 127.0.0.1:8765 -> build/mc_engine.exe -> GPU
```

- Hand-written CUDA kernels in `kernels/mc_risk.cu`. Pricing does not call cuRAND or a vendor pricing library.
- Philox4x32-10, checked against the published test vectors on the CPU and the GPU.
- European prices use antithetic variates and are checked against Black-Scholes.
- Portfolio scenarios use a Cholesky factor, then a full revaluation. VaR and expected shortfall come from a GPU histogram.
- One arithmetic Asian uses a geometric Asian control variate.
- A multi-threaded CPU baseline reports the speedup.
- The host finds `kernels/mc_risk.cu` and NVRTC from the executable location, `NVRTC_PATH`, or `CUDA_PATH`. Paths are not tied to a drive letter.

## Requirements

| Tool | Used for |
| --- | --- |
| 64-bit MinGW-w64 `gcc` | Windows host build. Set `CC` if it is not on `PATH`. |
| Python 3.9 or newer | Dashboard, job validation, and the NVRTC wheel install. |
| NVIDIA driver and GPU | Local runs. The build downloads the NVRTC redistributable into `.cuda_pkgs` when `nvcc` is not installed. |

## Quick start

From the repository root:

```powershell
powershell -File .\build.ps1
powershell -File .\run_dashboard.ps1
```

Open http://127.0.0.1:8765. The synthetic book can be run at 1 million, 10 million, or 100 million scenarios.

`build.ps1` resolves the repository from its own path. Compiler output, pip cache, and temporary files stay in `build\`, `.cache\`, and `tmp\` inside that repository. Those directories are gitignored.

Run the engine directly:

```powershell
.\build\mc_engine.exe selftest
.\build\mc_engine.exe run .\path\to\job.txt .\path\to\result.json
```

## Configuration

| Variable | Meaning |
| --- | --- |
| `CC` | 64-bit `gcc` used by `build.ps1`. |
| `MC_KERNEL` | Path to `mc_risk.cu`. Default: next to the repo, or beside the binary. |
| `NVRTC_PATH` | `nvrtc64_*.dll` or `libnvrtc.so.12`. Default: `.cuda_pkgs`, then `CUDA_PATH`. |
| `CUDA_PATH` | CUDA toolkit root. Its `bin` directory is searched for NVRTC. |

## Model

Prices follow geometric Brownian motion. European payoffs use antithetic normals and continuous discounting at the risk-free rate.

Each portfolio scenario applies the Cholesky factor of the correlation matrix, steps every underlying to the horizon, and revalues the book. Equities are marked at the simulated spot. Options are Black-Scholes with the remaining maturity. One trading day is `1/252` of a year.

VaR is the loss at the 95% and 99% P&L quantiles. Expected shortfall is the average P&L inside that tail. A negative VaR means that tail is still a gain.

## In the tree, not run

`aws/` holds a Lambda handler and a SAM template. `docker/` holds a Dockerfile and an entrypoint. `build.ps1` copies the handler into `lambda_pkg\`. No container was built, no stack was deployed, and no Batch job was submitted. The dashboard cost column multiplies the local GPU seconds by published on-demand list rates (`g4dn.xlarge` at `$0.526/hour`, `g5.xlarge` at `$1.006/hour`). That column is arithmetic in the page. It is not a measured cloud bill.

## Example results

These figures are from one workstation run of the synthetic book on an NVIDIA GeForce RTX 3050 Laptop. They are a reference, not a performance guarantee.

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
| `architecture.md` | Method, kernels, synthetic book, and measured local results |
| `kernels/mc_risk.cu` | RNG, European pricing, scenario VaR, Asian control variate |
| `src/engine.c` | Driver API host, CPU baseline, JSON writer |
| `dashboard/` | Browser and local API |
| `common/jobtext.py` | Synthetic sample book, validation, and the job text format |
| `aws/` | Lambda handler and SAM template. Present, not deployed |
| `docker/` | Dockerfile and entrypoint. Present, image not built |
| `build.ps1` | Locate a 64-bit compiler, fetch NVRTC if needed, build, self-test |
