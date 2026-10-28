/*
 * Monte Carlo risk engine — device code.
 *
 * Kernels (no library pricing, no cuRAND):
 *   Philox4x32-10 counter-based RNG (Random123 round function)
 *   European option pricing with antithetic variates
 *   Correlated GBM portfolio scenarios, full Black-Scholes revaluation
 *   Histogram of P&L for Value-at-Risk and Expected Shortfall
 *   Arithmetic Asian call with a geometric control variate
 *
 * Compiled at runtime with NVRTC, or ahead of time with nvcc.
 * The source is self-contained: NVRTC gets explicit device declarations,
 * nvcc already provides those builtins.
 */

#ifdef __CUDACC_RTC__
__device__ double exp(double);
__device__ double log(double);
__device__ double sqrt(double);
__device__ double cos(double);
__device__ double sin(double);
__device__ double erfc(double);
__device__ int atomicAdd(int*, int);
__device__ unsigned long long atomicAdd(unsigned long long*, unsigned long long);
__device__ unsigned long long atomicCAS(unsigned long long*, unsigned long long, unsigned long long);
#endif

__device__ double atomic_add_double(double* addr, double val) {
    unsigned long long* ptr = (unsigned long long*)addr;
    unsigned long long old = *ptr;
    unsigned long long assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(ptr, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double dmax(double a, double b) { return a > b ? a : b; }

__device__ double norm_cdf(double x) {
    return 0.5 * erfc(-x * 0.70710678118654752440);
}

__device__ double bs_price(double s, double k, double r, double q, double vol, double t, int is_call) {
    if (t <= 1e-14) {
        double intrinsic = is_call ? dmax(s - k, 0.0) : dmax(k - s, 0.0);
        return intrinsic;
    }
    if (vol <= 1e-14) {
        double forward = s * exp((r - q) * t);
        double df = exp(-r * t);
        double undisc = is_call ? dmax(forward - k, 0.0) : dmax(k - forward, 0.0);
        return df * undisc;
    }
    double root = sqrt(t);
    double sdt = vol * root;
    double d1 = (log(s / k) + (r - q + 0.5 * vol * vol) * t) / sdt;
    double d2 = d1 - sdt;
    double df_r = exp(-r * t);
    double df_q = exp(-q * t);
    if (is_call)
        return s * df_q * norm_cdf(d1) - k * df_r * norm_cdf(d2);
    return k * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1);
}

/* Philox4x32-10. Counter (c0,c1,c2,c3), key (k0,k1). */
__device__ void philox4x32_10(unsigned int c0, unsigned int c1, unsigned int c2, unsigned int c3,
                              unsigned int k0, unsigned int k1, unsigned int out[4]) {
    unsigned int c[4];
    c[0] = c0;
    c[1] = c1;
    c[2] = c2;
    c[3] = c3;
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = (unsigned long long)0xD2511F53u * (unsigned long long)c[0];
        unsigned long long p1 = (unsigned long long)0xCD9E8D57u * (unsigned long long)c[2];
        unsigned int lo0 = (unsigned int)p0;
        unsigned int hi0 = (unsigned int)(p0 >> 32);
        unsigned int lo1 = (unsigned int)p1;
        unsigned int hi1 = (unsigned int)(p1 >> 32);
        unsigned int n0 = hi1 ^ c[1] ^ k0;
        unsigned int n1 = lo1;
        unsigned int n2 = hi0 ^ c[3] ^ k1;
        unsigned int n3 = lo0;
        c[0] = n0;
        c[1] = n1;
        c[2] = n2;
        c[3] = n3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
    out[3] = c[3];
}

__device__ void box_muller(unsigned int ua, unsigned int ub, double* z0, double* z1) {
    double u = ((double)ua + 0.5) * (1.0 / 4294967296.0);
    double v = ((double)ub + 0.5) * (1.0 / 4294967296.0);
    double radius = sqrt(-2.0 * log(u));
    double theta = 6.28318530717958647692 * v;
    *z0 = radius * cos(theta);
    *z1 = radius * sin(theta);
}

__device__ double block_sum(double value) {
    __shared__ double sm[256];
    unsigned int tid = threadIdx.x;
    sm[tid] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    return sm[0];
}

extern "C" __global__ void philox_kat(unsigned int* out4) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int o[4];
        philox4x32_10(0u, 0u, 0u, 0u, 0u, 0u, o);
        out4[0] = o[0];
        out4[1] = o[1];
        out4[2] = o[2];
        out4[3] = o[3];
        philox4x32_10(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, o);
        out4[4] = o[0];
        out4[5] = o[1];
        out4[6] = o[2];
        out4[7] = o[3];
    }
}

extern "C" __global__ void bs_one(double s, double k, double r, double q, double vol, double t,
                                  int is_call, double* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *out = bs_price(s, k, r, q, vol, t, is_call);
}

/*
 * One block-column per option (blockIdx.x). blockIdx.y stripes the pairs.
 * n_paths is the number of scenarios; they are consumed as antithetic pairs.
 */
extern "C" __global__ void price_options(
    const double* spots, const double* strikes, const double* rates, const double* divs,
    const double* vols, const double* mats, const int* is_call, int n_opt,
    unsigned long long n_paths, unsigned long long seed, double* sum, double* sumsq) {
    int opt = (int)blockIdx.x;
    if (opt >= n_opt)
        return;

    unsigned long long n_pairs = n_paths >> 1;
    unsigned long long stride = (unsigned long long)gridDim.y * blockDim.x;
    unsigned long long start = (unsigned long long)blockIdx.y * blockDim.x + threadIdx.x;

    double s0 = spots[opt];
    double k = strikes[opt];
    double r = rates[opt];
    double q = divs[opt];
    double vol = vols[opt];
    double t = mats[opt];
    int call = is_call[opt];
    double drift = (r - q - 0.5 * vol * vol) * t;
    double scale = vol * sqrt(t);
    double disc = exp(-r * t);
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);

    double local = 0.0;
    double local2 = 0.0;
    for (unsigned long long p = start; p < n_pairs; p += stride) {
        unsigned int o[4];
        philox4x32_10((unsigned int)p, (unsigned int)(p >> 32), (unsigned int)opt, 1u, k0, k1, o);
        double z, z_unused;
        box_muller(o[0], o[1], &z, &z_unused);
        double st_hi = s0 * exp(drift + scale * z);
        double st_lo = s0 * exp(drift - scale * z);
        double pay_hi = call ? dmax(st_hi - k, 0.0) : dmax(k - st_hi, 0.0);
        double pay_lo = call ? dmax(st_lo - k, 0.0) : dmax(k - st_lo, 0.0);
        double avg = 0.5 * disc * (pay_hi + pay_lo);
        local += avg;
        local2 += avg * avg;
    }

    double tot = block_sum(local);
    double tot2 = block_sum(local2);
    if (threadIdx.x == 0) {
        atomic_add_double(sum + opt, tot);
        atomic_add_double(sumsq + opt, tot2);
    }
}

__device__ double portfolio_value(
    const double* spot_now, double time_elapsed,
    const double* rates, const double* divs, const double* vols,
    const int* kinds, const int* und_idx, const double* strikes, const double* mats,
    const double* qtys, int n_pos) {
    double value = 0.0;
    for (int p = 0; p < n_pos; ++p) {
        int u = und_idx[p];
        double s = spot_now[u];
        double qty = qtys[p];
        int kind = kinds[p];
        if (kind == 0) {
            value += qty * s;
            continue;
        }
        double remain = mats[p] - time_elapsed;
        if (remain <= 1e-14) {
            double payoff = (kind == 1) ? dmax(s - strikes[p], 0.0) : dmax(strikes[p] - s, 0.0);
            value += qty * payoff;
        } else {
            value += qty * bs_price(s, strikes[p], rates[u], divs[u], vols[u], remain, kind == 1);
        }
    }
    return value;
}

__device__ double scenario_pnl(
    unsigned long long path, unsigned long long seed, double v0,
    const double* spots, const double* vols, const double* rates, const double* divs,
    const double* drifts, const double* chol, int n_und,
    const int* kinds, const int* und_idx, const double* strikes, const double* mats,
    const double* qtys, int n_pos, double horizon) {
    unsigned int o[4];
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);
    double z[16];
    int filled = 0;
    int pair = 0;
    while (filled < n_und) {
        philox4x32_10((unsigned int)path, (unsigned int)(path >> 32), (unsigned int)pair, 2u, k0, k1, o);
        double n0, n1, n2, n3;
        box_muller(o[0], o[1], &n0, &n1);
        box_muller(o[2], o[3], &n2, &n3);
        if (filled < n_und) z[filled++] = n0;
        if (filled < n_und) z[filled++] = n1;
        if (filled < n_und) z[filled++] = n2;
        if (filled < n_und) z[filled++] = n3;
        ++pair;
    }

    double shocked[16];
    double root_dt = sqrt(horizon);
    for (int i = 0; i < n_und; ++i) {
        double y = 0.0;
        for (int j = 0; j <= i; ++j)
            y += chol[i * n_und + j] * z[j];
        double vol = vols[i];
        double mu = drifts[i];
        shocked[i] = spots[i] * exp((mu - 0.5 * vol * vol) * horizon + vol * root_dt * y);
    }

    double vt = portfolio_value(shocked, horizon, rates, divs, vols, kinds, und_idx, strikes, mats, qtys, n_pos);
    return vt - v0;
}

extern "C" __global__ void book_value(
    const double* spots, const double* rates, const double* divs, const double* vols,
    const int* kinds, const int* und_idx, const double* strikes, const double* mats,
    const double* qtys, int n_pos, double* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = portfolio_value(spots, 0.0, rates, divs, vols, kinds, und_idx, strikes, mats, qtys, n_pos);
    }
}

extern "C" __global__ void var_moments(
    const double* spots, const double* vols, const double* rates, const double* divs,
    const double* drifts, const double* chol, int n_und,
    const int* kinds, const int* und_idx, const double* strikes, const double* mats,
    const double* qtys, int n_pos, double horizon,
    unsigned long long n_paths, unsigned long long seed,
    double* sum_pnl, double* sum_pnl_sq) {
    double v0 = portfolio_value(spots, 0.0, rates, divs, vols, kinds, und_idx, strikes, mats, qtys, n_pos);
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long start = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    double local2 = 0.0;
    for (unsigned long long p = start; p < n_paths; p += stride) {
        double pnl = scenario_pnl(p, seed, v0, spots, vols, rates, divs, drifts, chol, n_und,
                                   kinds, und_idx, strikes, mats, qtys, n_pos, horizon);
        local += pnl;
        local2 += pnl * pnl;
    }
    double tot = block_sum(local);
    double tot2 = block_sum(local2);
    if (threadIdx.x == 0) {
        atomic_add_double(sum_pnl, tot);
        atomic_add_double(sum_pnl_sq, tot2);
    }
}

extern __shared__ unsigned int g_shist[];

extern "C" __global__ void var_hist(
    const double* spots, const double* vols, const double* rates, const double* divs,
    const double* drifts, const double* chol, int n_und,
    const int* kinds, const int* und_idx, const double* strikes, const double* mats,
    const double* qtys, int n_pos, double horizon,
    unsigned long long n_paths, unsigned long long seed,
    double lo, double hi, int nbins,
    unsigned long long* hist, unsigned long long* under, unsigned long long* over) {
    for (int i = (int)threadIdx.x; i < nbins; i += (int)blockDim.x)
        g_shist[i] = 0u;
    __syncthreads();

    double v0 = portfolio_value(spots, 0.0, rates, divs, vols, kinds, und_idx, strikes, mats, qtys, n_pos);
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long start = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    double width = (hi - lo) / (double)nbins;
    unsigned int local_under = 0u;
    unsigned int local_over = 0u;

    for (unsigned long long p = start; p < n_paths; p += stride) {
        double pnl = scenario_pnl(p, seed, v0, spots, vols, rates, divs, drifts, chol, n_und,
                                   kinds, und_idx, strikes, mats, qtys, n_pos, horizon);
        int bin = (int)((pnl - lo) / width);
        if (bin < 0)
            ++local_under;
        else if (bin >= nbins)
            ++local_over;
        else
            atomicAdd((int*)&g_shist[bin], 1);
    }

    __syncthreads();
    for (int i = (int)threadIdx.x; i < nbins; i += (int)blockDim.x) {
        unsigned int c = g_shist[i];
        if (c)
            atomicAdd(hist + i, (unsigned long long)c);
    }
    if (local_under)
        atomicAdd(under, (unsigned long long)local_under);
    if (local_over)
        atomicAdd(over, (unsigned long long)local_over);
}

extern "C" __global__ void asian_mc(
    double spot, double strike, double rate, double divd, double vol, double mat, int steps,
    unsigned long long n_paths, unsigned long long seed,
    double* sum_a, double* sum_g, double* sum_a2, double* sum_g2, double* sum_ag) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long start = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    double dt = mat / (double)steps;
    double nudt = (rate - divd - 0.5 * vol * vol) * dt;
    double sigdt = vol * sqrt(dt);
    double disc = exp(-rate * mat);
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);

    double local_a = 0.0, local_g = 0.0, local_a2 = 0.0, local_g2 = 0.0, local_ag = 0.0;
    for (unsigned long long p = start; p < n_paths; p += stride) {
        double log_s = log(spot);
        double sum_s = 0.0;
        double sum_log = 0.0;
        double pending[4];
        int pending_n = 0;
        int pair = 0;
        for (int step = 0; step < steps; ++step) {
            if (pending_n == 0) {
                unsigned int o[4];
                philox4x32_10((unsigned int)p, (unsigned int)(p >> 32), (unsigned int)pair, 3u, k0, k1, o);
                box_muller(o[0], o[1], &pending[0], &pending[1]);
                box_muller(o[2], o[3], &pending[2], &pending[3]);
                pending_n = 4;
                ++pair;
            }
            double z = pending[4 - pending_n];
            --pending_n;
            log_s += nudt + sigdt * z;
            sum_s += exp(log_s);
            sum_log += log_s;
        }
        double arith = sum_s / (double)steps;
        double geo = exp(sum_log / (double)steps);
        double pa = disc * dmax(arith - strike, 0.0);
        double pg = disc * dmax(geo - strike, 0.0);
        local_a += pa;
        local_g += pg;
        local_a2 += pa * pa;
        local_g2 += pg * pg;
        local_ag += pa * pg;
    }

    double ta = block_sum(local_a);
    double tg = block_sum(local_g);
    double ta2 = block_sum(local_a2);
    double tg2 = block_sum(local_g2);
    double tag = block_sum(local_ag);
    if (threadIdx.x == 0) {
        atomic_add_double(sum_a, ta);
        atomic_add_double(sum_g, tg);
        atomic_add_double(sum_a2, ta2);
        atomic_add_double(sum_g2, tg2);
        atomic_add_double(sum_ag, tag);
    }
}
