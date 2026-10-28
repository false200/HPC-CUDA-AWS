/*
 * Host for the CUDA Monte Carlo risk engine.
 * Loads the NVIDIA driver and NVRTC at runtime, compiles kernels/mc_risk.cu,
 * and writes a JSON result. The CPU path is the speedup baseline.
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif
#else
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#endif

#define MAX_U 16
#define MAX_P 64
#define NBINS 4096
#define DISP_BINS 128
#define BLOCK 256

_Static_assert(sizeof(void*) == 8, "build the host as 64-bit so it matches the GPU driver");
_Static_assert(NBINS % DISP_BINS == 0, "display bins must divide the histogram");

typedef unsigned long long ull;

static char g_err[4096];

static void set_err(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

static double qpc_origin;

static double now_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / qpc_origin;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void time_init(void) {
#ifdef _WIN32
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    qpc_origin = (double)f.QuadPart;
#else
    qpc_origin = 1.0;
#endif
}

/* ---------- Philox4x32-10 (same round as the device) ---------- */

static void philox4x32_10(unsigned c0, unsigned c1, unsigned c2, unsigned c3,
                          unsigned k0, unsigned k1, unsigned out[4]) {
    unsigned c[4] = {c0, c1, c2, c3};
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = (unsigned long long)0xD2511F53u * (unsigned long long)c[0];
        unsigned long long p1 = (unsigned long long)0xCD9E8D57u * (unsigned long long)c[2];
        unsigned lo0 = (unsigned)p0, hi0 = (unsigned)(p0 >> 32);
        unsigned lo1 = (unsigned)p1, hi1 = (unsigned)(p1 >> 32);
        unsigned n0 = hi1 ^ c[1] ^ k0;
        unsigned n1 = lo1;
        unsigned n2 = hi0 ^ c[3] ^ k1;
        unsigned n3 = lo0;
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

static int kat_cpu(void) {
    unsigned o[4];
    static const unsigned zexp[4] = {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u};
    static const unsigned oexp[4] = {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu};
    philox4x32_10(0, 0, 0, 0, 0, 0, o);
    for (int i = 0; i < 4; ++i)
        if (o[i] != zexp[i]) return 0;
    philox4x32_10(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, o);
    for (int i = 0; i < 4; ++i)
        if (o[i] != oexp[i]) return 0;
    return 1;
}

static void box_muller(unsigned ua, unsigned ub, double* z0, double* z1) {
    double u = ((double)ua + 0.5) * (1.0 / 4294967296.0);
    double v = ((double)ub + 0.5) * (1.0 / 4294967296.0);
    double radius = sqrt(-2.0 * log(u));
    double theta = 6.28318530717958647692 * v;
    *z0 = radius * cos(theta);
    *z1 = radius * sin(theta);
}

static double norm_cdf(double x) { return 0.5 * erfc(-x * 0.70710678118654752440); }

static double norm_pdf(double x) {
    return exp(-0.5 * x * x) * 0.39894228040143267794;
}

static double norm_ppf(double p) {
    double lo = -12.0, hi = 12.0;
    for (int i = 0; i < 80; ++i) {
        double mid = 0.5 * (lo + hi);
        if (norm_cdf(mid) < p) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

static double bs_price(double s, double k, double r, double q, double vol, double t, int is_call) {
    if (t <= 1e-14) {
        double intrinsic = is_call ? fmax(s - k, 0.0) : fmax(k - s, 0.0);
        return intrinsic;
    }
    if (vol <= 1e-14) {
        double forward = s * exp((r - q) * t);
        double df = exp(-r * t);
        double undisc = is_call ? fmax(forward - k, 0.0) : fmax(k - forward, 0.0);
        return df * undisc;
    }
    double sdt = vol * sqrt(t);
    double d1 = (log(s / k) + (r - q + 0.5 * vol * vol) * t) / sdt;
    double d2 = d1 - sdt;
    double df_r = exp(-r * t);
    double df_q = exp(-q * t);
    if (is_call) return s * df_q * norm_cdf(d1) - k * df_r * norm_cdf(d2);
    return k * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1);
}

static void bs_greeks(double s, double k, double r, double q, double vol, double t, int is_call,
                      double* price, double* delta, double* vega) {
    *price = bs_price(s, k, r, q, vol, t, is_call);
    if (t <= 1e-14 || vol <= 1e-14) {
        double intrinsic_up = is_call ? (s >= k) : (s <= k);
        *delta = is_call ? (s > k ? exp(-q * t) : 0.0) : (s < k ? -exp(-q * t) : 0.0);
        if (fabs(s - k) < 1e-12) *delta = is_call ? 0.5 * exp(-q * t) : -0.5 * exp(-q * t);
        (void)intrinsic_up;
        *vega = 0.0;
        return;
    }
    double sdt = vol * sqrt(t);
    double d1 = (log(s / k) + (r - q + 0.5 * vol * vol) * t) / sdt;
    double df_q = exp(-q * t);
    *delta = is_call ? df_q * norm_cdf(d1) : df_q * (norm_cdf(d1) - 1.0);
    *vega = s * df_q * norm_pdf(d1) * sqrt(t);
}

static double geo_asian_call(double s, double k, double r, double q, double vol, double t, int steps) {
    double n = (double)steps;
    double m = log(s) + (r - q - 0.5 * vol * vol) * t * (n + 1.0) / (2.0 * n);
    double v = vol * vol * t * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
    double root = sqrt(v);
    double d2 = (m - log(k)) / root;
    double d1 = d2 + root;
    return exp(-r * t) * (exp(m + 0.5 * v) * norm_cdf(d1) - k * norm_cdf(d2));
}

/* ---------- job ---------- */

typedef struct {
    char name[32];
    double spot, vol, rate, div, drift;
} Und;

typedef struct {
    int und;
    int kind; /* 0 equity, 1 call, 2 put */
    double strike, mat, qty;
} Pos;

typedef struct {
    ull seed, paths, bench_paths;
    double horizon_days;
    int n_und, n_pos;
    int has_asian, asian_und, asian_steps;
    double asian_k, asian_mat, asian_qty;
    Und und[MAX_U];
    double corr[MAX_U * MAX_U];
    Pos pos[MAX_P];
} Job;

static int cholesky(const double* corr, double* L, int n) {
    memset(L, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = corr[i * n + j];
            for (int k = 0; k < j; ++k) sum -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (sum <= 1e-12) {
                    set_err("correlation matrix is not positive definite");
                    return 0;
                }
                L[i * n + j] = sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }
    return 1;
}

static int tokenize(char* buf, char** toks, int cap, int* nout) {
    int n = 0;
    char* p = buf;
    if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF)
        p += 3;
    while (*p) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
        if (!*p) break;
        if (n >= cap) {
            set_err("job file is too large");
            return 0;
        }
        toks[n++] = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') ++p;
        if (*p) {
            *p = 0;
            ++p;
        }
    }
    *nout = n;
    return 1;
}

static int need_key(char** t, int n, int* i, const char* key) {
    if (*i >= n || strcmp(t[*i], key) != 0) {
        set_err("expected '%s' in the job file", key);
        return 0;
    }
    ++(*i);
    return 1;
}

static int read_f(char** t, int n, int* i, double* o) {
    if (*i >= n) {
        set_err("job file ended early");
        return 0;
    }
    char* end = NULL;
    *o = strtod(t[*i], &end);
    if (end == t[*i] || *end != 0 || !isfinite(*o)) {
        set_err("bad number '%s'", t[*i]);
        return 0;
    }
    ++(*i);
    return 1;
}

static int read_i(char** t, int n, int* i, int* o) {
    double v;
    if (!read_f(t, n, i, &v)) return 0;
    if (v < -2e9 || v > 2e9 || v != floor(v)) {
        set_err("expected an integer");
        return 0;
    }
    *o = (int)v;
    return 1;
}

static int read_ull(char** t, int n, int* i, ull* o) {
    if (*i >= n) {
        set_err("job file ended early");
        return 0;
    }
    char* end = NULL;
    unsigned long long v = strtoull(t[*i], &end, 10);
    if (end == t[*i] || *end != 0) {
        set_err("bad integer '%s'", t[*i]);
        return 0;
    }
    *o = v;
    ++(*i);
    return 1;
}

static int parse_job(char* buf, Job* job) {
    char* toks[4096];
    int n = 0, i = 0;
    memset(job, 0, sizeof *job);
    if (!tokenize(buf, toks, 4096, &n)) return 0;
    if (i >= n || strcmp(toks[i++], "MCJOB") != 0) {
        set_err("job must start with MCJOB 1");
        return 0;
    }
    int ver = 0;
    if (!read_i(toks, n, &i, &ver) || ver != 1) {
        set_err("unsupported job version");
        return 0;
    }
    if (!need_key(toks, n, &i, "seed") || !read_ull(toks, n, &i, &job->seed)) return 0;
    if (!need_key(toks, n, &i, "paths") || !read_ull(toks, n, &i, &job->paths)) return 0;
    if (!need_key(toks, n, &i, "bench_paths") || !read_ull(toks, n, &i, &job->bench_paths)) return 0;
    if (!need_key(toks, n, &i, "horizon_days") || !read_f(toks, n, &i, &job->horizon_days)) return 0;
    if (!need_key(toks, n, &i, "underlyings") || !read_i(toks, n, &i, &job->n_und)) return 0;
    if (job->n_und < 1 || job->n_und > MAX_U) {
        set_err("underlyings must be between 1 and %d", MAX_U);
        return 0;
    }
    if (job->paths < 10000ull || job->paths > 200000000ull) {
        set_err("paths must be between 10000 and 200000000");
        return 0;
    }
    if (job->bench_paths < 10000ull || job->bench_paths > job->paths) {
        set_err("bench_paths must be between 10000 and paths");
        return 0;
    }
    if (!(job->horizon_days > 0.0) || job->horizon_days > 252.0) {
        set_err("horizon_days must be in (0, 252]");
        return 0;
    }
    double horizon = job->horizon_days / 252.0;
    for (int u = 0; u < job->n_und; ++u) {
        if (i >= n) {
            set_err("missing underlying");
            return 0;
        }
        if (strlen(toks[i]) > 31) {
            set_err("underlying name is too long");
            return 0;
        }
        memcpy(job->und[u].name, toks[i], strlen(toks[i]) + 1);
        ++i;
        if (!read_f(toks, n, &i, &job->und[u].spot) || !read_f(toks, n, &i, &job->und[u].vol) ||
            !read_f(toks, n, &i, &job->und[u].rate) || !read_f(toks, n, &i, &job->und[u].div) ||
            !read_f(toks, n, &i, &job->und[u].drift))
            return 0;
        if (!(job->und[u].spot > 0.0) || !(job->und[u].vol > 0.0)) {
            set_err("%s spot and vol must be positive", job->und[u].name);
            return 0;
        }
    }
    if (!need_key(toks, n, &i, "correlation")) return 0;
    for (int k = 0; k < job->n_und * job->n_und; ++k) {
        if (!read_f(toks, n, &i, &job->corr[k])) return 0;
    }
    for (int r = 0; r < job->n_und; ++r) {
        if (fabs(job->corr[r * job->n_und + r] - 1.0) > 1e-6) {
            set_err("correlation diagonal must be 1");
            return 0;
        }
        for (int c = 0; c < r; ++c) {
            if (fabs(job->corr[r * job->n_und + c] - job->corr[c * job->n_und + r]) > 1e-6) {
                set_err("correlation matrix must be symmetric");
                return 0;
            }
        }
    }
    if (!need_key(toks, n, &i, "positions") || !read_i(toks, n, &i, &job->n_pos)) return 0;
    if (job->n_pos < 1 || job->n_pos > MAX_P) {
        set_err("positions must be between 1 and %d", MAX_P);
        return 0;
    }
    for (int p = 0; p < job->n_pos; ++p) {
        if (!read_i(toks, n, &i, &job->pos[p].und)) return 0;
        if (i >= n) {
            set_err("position ended early");
            return 0;
        }
        if (strcmp(toks[i], "equity") == 0) job->pos[p].kind = 0;
        else if (strcmp(toks[i], "call") == 0) job->pos[p].kind = 1;
        else if (strcmp(toks[i], "put") == 0) job->pos[p].kind = 2;
        else {
            set_err("unknown position kind '%s'", toks[i]);
            return 0;
        }
        ++i;
        if (!read_f(toks, n, &i, &job->pos[p].strike) || !read_f(toks, n, &i, &job->pos[p].mat) ||
            !read_f(toks, n, &i, &job->pos[p].qty))
            return 0;
        if (job->pos[p].und < 0 || job->pos[p].und >= job->n_und) {
            set_err("position underlying index is out of range");
            return 0;
        }
        if (job->pos[p].qty == 0.0) {
            set_err("position quantity must be non-zero");
            return 0;
        }
        if (job->pos[p].kind != 0) {
            if (!(job->pos[p].strike > 0.0) || !(job->pos[p].mat > horizon)) {
                set_err("option maturity must be longer than the VaR horizon");
                return 0;
            }
        }
    }
    if (i < n) {
        if (!need_key(toks, n, &i, "asian")) return 0;
        job->has_asian = 1;
        if (!read_i(toks, n, &i, &job->asian_und) || !read_f(toks, n, &i, &job->asian_k) ||
            !read_f(toks, n, &i, &job->asian_mat) || !read_i(toks, n, &i, &job->asian_steps) ||
            !read_f(toks, n, &i, &job->asian_qty))
            return 0;
        if (job->asian_und < 0 || job->asian_und >= job->n_und || !(job->asian_k > 0.0) ||
            !(job->asian_mat > 0.0) || job->asian_steps < 2 || job->asian_steps > 256 || job->asian_qty == 0.0) {
            set_err("asian parameters are out of range");
            return 0;
        }
    }
    if (i != n) {
        set_err("unexpected trailing tokens in the job file");
        return 0;
    }
    return 1;
}

static int parse_file(const char* path, Job* job) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        set_err("cannot open %s", path);
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        set_err("cannot read %s", path);
        return 0;
    }
    long sz = ftell(f);
    if (sz < 0 || sz > 1000000) {
        fclose(f);
        set_err("job file size is not usable");
        return 0;
    }
    rewind(f);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        set_err("out of memory");
        return 0;
    }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = 0;
    int ok = parse_job(buf, job);
    free(buf);
    return ok;
}

/* ---------- CUDA loader ---------- */

typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;
typedef int CUdevice;

typedef int (*fn_cuInit)(unsigned);
typedef int (*fn_cuDeviceGet)(CUdevice*, int);
typedef int (*fn_cuDeviceGetName)(char*, int, CUdevice);
typedef int (*fn_cuDeviceGetAttribute)(int*, int, CUdevice);
typedef int (*fn_cuCtxCreate)(CUcontext*, unsigned, CUdevice);
typedef int (*fn_cuCtxSynchronize)(void);
typedef int (*fn_cuModuleLoadData)(CUmodule*, const void*);
typedef int (*fn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef int (*fn_cuMemAlloc)(CUdeviceptr*, size_t);
typedef int (*fn_cuMemFree)(CUdeviceptr);
typedef int (*fn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef int (*fn_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef int (*fn_cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
typedef int (*fn_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                 unsigned, void*, void**, void**);
typedef int (*fn_cuGetErrorName)(int, const char**);

typedef struct nvrtcProgram_st* nvrtcProgram;
typedef int (*fn_nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*,
                                     const char* const*);
typedef int (*fn_nvrtcDestroyProgram)(nvrtcProgram*);
typedef int (*fn_nvrtcCompileProgram)(nvrtcProgram, int, const char* const*);
typedef int (*fn_nvrtcGetPTXSize)(nvrtcProgram, size_t*);
typedef int (*fn_nvrtcGetPTX)(nvrtcProgram, char*);
typedef int (*fn_nvrtcGetCUBINSize)(nvrtcProgram, size_t*);
typedef int (*fn_nvrtcGetCUBIN)(nvrtcProgram, char*);
typedef int (*fn_nvrtcGetProgramLogSize)(nvrtcProgram, size_t*);
typedef int (*fn_nvrtcGetProgramLog)(nvrtcProgram, char*);

static fn_cuInit p_cuInit;
static fn_cuDeviceGet p_cuDeviceGet;
static fn_cuDeviceGetName p_cuDeviceGetName;
static fn_cuDeviceGetAttribute p_cuDeviceGetAttribute;
static fn_cuCtxCreate p_cuCtxCreate;
static fn_cuCtxSynchronize p_cuCtxSynchronize;
static fn_cuModuleLoadData p_cuModuleLoadData;
static fn_cuModuleGetFunction p_cuModuleGetFunction;
static fn_cuMemAlloc p_cuMemAlloc;
static fn_cuMemFree p_cuMemFree;
static fn_cuMemcpyHtoD p_cuMemcpyHtoD;
static fn_cuMemcpyDtoH p_cuMemcpyDtoH;
static fn_cuMemsetD8 p_cuMemsetD8;
static fn_cuLaunchKernel p_cuLaunchKernel;
static fn_cuGetErrorName p_cuGetErrorName;

static CUfunction f_kat, f_bs, f_price, f_book, f_moments, f_hist, f_asian;
static int g_ready = 0;
static int g_sms = 1;
static int g_major = 0, g_minor = 0, g_clock_khz = 0;
static int g_from_cache = 0;
static double g_compile_seconds = 0.0;
static char g_gpu_name[256] = "GPU";
static char g_cache_dir[1024];

static int cu_ok(int err, const char* what) {
    if (err == 0) return 1;
    const char* name = NULL;
    if (p_cuGetErrorName) p_cuGetErrorName(err, &name);
    set_err("%s failed (%d%s%s)", what, err, name ? " " : "", name ? name : "");
    return 0;
}

static void* lib_sym(void* lib, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static int load_driver(void) {
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
    if (!lib) {
        set_err("nvcuda.dll is not available. The NVIDIA driver is required.");
        return 0;
    }
#else
    void* lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libcuda.so", RTLD_NOW);
    if (!lib) {
        set_err("libcuda is not available. The NVIDIA driver is required.");
        return 0;
    }
#endif
#define LOAD(field, name)                                          \
    do {                                                           \
        p_##field = (fn_##field)lib_sym(lib, name);                \
        if (!p_##field) {                                          \
            set_err("driver is missing %s", name);                 \
            return 0;                                              \
        }                                                          \
    } while (0)
    LOAD(cuInit, "cuInit");
    LOAD(cuDeviceGet, "cuDeviceGet");
    LOAD(cuDeviceGetName, "cuDeviceGetName");
    LOAD(cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD(cuCtxCreate, "cuCtxCreate_v2");
    LOAD(cuCtxSynchronize, "cuCtxSynchronize");
    LOAD(cuModuleLoadData, "cuModuleLoadData");
    LOAD(cuModuleGetFunction, "cuModuleGetFunction");
    LOAD(cuMemAlloc, "cuMemAlloc_v2");
    LOAD(cuMemFree, "cuMemFree_v2");
    LOAD(cuMemcpyHtoD, "cuMemcpyHtoD_v2");
    LOAD(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
    LOAD(cuMemsetD8, "cuMemsetD8_v2");
    LOAD(cuLaunchKernel, "cuLaunchKernel");
#undef LOAD
    p_cuGetErrorName = (fn_cuGetErrorName)lib_sym(lib, "cuGetErrorName");
    return 1;
}

static ull fnv64(const char* s, size_t n) {
    ull h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static char* read_file(const char* path, size_t* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0 || sz > 4000000) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = 0;
    *out_n = got;
    return buf;
}

static void exe_dir(char* out, size_t n) {
    char buf[1024];
    buf[0] = 0;
#ifdef _WIN32
    DWORD k = GetModuleFileNameA(NULL, buf, (DWORD)sizeof buf);
    if (k == 0 || k >= sizeof buf) buf[0] = 0;
#else
    ssize_t k = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (k < 0) k = 0;
    buf[k] = 0;
#endif
    char* slash = strrchr(buf, '\\');
    char* slash2 = strrchr(buf, '/');
    if (!slash || (slash2 && slash2 > slash)) slash = slash2;
    if (slash) *slash = 0;
    snprintf(out, n, "%s", buf[0] ? buf : ".");
}

static void parent_of(const char* dir, char* out, size_t n) {
    snprintf(out, n, "%s", dir);
    char* slash = strrchr(out, '\\');
    char* slash2 = strrchr(out, '/');
    if (!slash || (slash2 && slash2 > slash)) slash = slash2;
    if (slash) *slash = 0;
    else snprintf(out, n, ".");
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void path_join(char* out, size_t n, const char* a, const char* b) {
#ifdef _WIN32
    snprintf(out, n, "%s\\%s", a, b);
#else
    snprintf(out, n, "%s/%s", a, b);
#endif
}

static void find_kernel(char* out, size_t n) {
    const char* env = getenv("MC_KERNEL");
    if (env && env[0] && file_exists(env)) {
        snprintf(out, n, "%s", env);
        return;
    }
    char dir[1024], parent[1024], cand[1200];
    exe_dir(dir, sizeof dir);
    parent_of(dir, parent, sizeof parent);
    path_join(cand, sizeof cand, parent, "kernels");
    path_join(out, n, cand, "mc_risk.cu");
    if (file_exists(out)) return;
    path_join(cand, sizeof cand, dir, "kernels");
    path_join(out, n, cand, "mc_risk.cu");
    if (file_exists(out)) return;
    snprintf(out, n, "kernels%cmc_risk.cu",
#ifdef _WIN32
             '\\'
#else
             '/'
#endif
    );
}

static int load_module_bytes(const void* image, CUmodule* mod) {
    return cu_ok(p_cuModuleLoadData(mod, image), "cuModuleLoadData");
}

static int bind_functions(CUmodule mod) {
    struct {
        CUfunction* fn;
        const char* name;
    } items[] = {{&f_kat, "philox_kat"},     {&f_bs, "bs_one"},         {&f_price, "price_options"},
                 {&f_book, "book_value"},    {&f_moments, "var_moments"}, {&f_hist, "var_hist"},
                 {&f_asian, "asian_mc"}};
    for (size_t i = 0; i < sizeof items / sizeof items[0]; ++i) {
        if (!cu_ok(p_cuModuleGetFunction(items[i].fn, mod, items[i].name), items[i].name)) return 0;
    }
    return 1;
}

static int save_cache(const char* path, const void* bytes, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    size_t w = fwrite(bytes, 1, n, f);
    fclose(f);
    return w == n;
}

#ifdef _WIN32
static int find_in_dir(const char* dir, const char* pattern, char* out, size_t n) {
    char spec[1100];
    WIN32_FIND_DATAA fd;
    HANDLE handle;
    snprintf(spec, sizeof spec, "%s\\%s", dir, pattern);
    handle = FindFirstFileA(spec, &fd);
    if (handle == INVALID_HANDLE_VALUE) return 0;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        snprintf(out, n, "%s\\%s", dir, fd.cFileName);
        FindClose(handle);
        return 1;
    } while (FindNextFileA(handle, &fd));
    FindClose(handle);
    return 0;
}

static int locate_nvrtc(char* out, size_t n) {
    const char* env = getenv("NVRTC_PATH");
    char exe[1024], parent[1024], dir[1100];
    const char* cuda;
    const char* rels[] = {".cuda_pkgs\\nvidia\\cuda_nvrtc\\bin", ".cuda_pkgs\\nvidia\\cu12\\bin\\x86_64", "bin"};
    char roots[2][1024];
    int r, i;
    if (env && env[0] && file_exists(env)) {
        snprintf(out, n, "%s", env);
        return 1;
    }
    exe_dir(exe, sizeof exe);
    parent_of(exe, parent, sizeof parent);
    snprintf(roots[0], sizeof roots[0], "%s", parent);
    snprintf(roots[1], sizeof roots[1], "%s", exe);
    for (r = 0; r < 2; ++r) {
        for (i = 0; i < 3; ++i) {
            snprintf(dir, sizeof dir, "%s\\%s", roots[r], rels[i]);
            if (find_in_dir(dir, "nvrtc64_*.dll", out, n)) return 1;
        }
    }
    cuda = getenv("CUDA_PATH");
    if (cuda && cuda[0]) {
        snprintf(dir, sizeof dir, "%s\\bin", cuda);
        if (find_in_dir(dir, "nvrtc64_*.dll", out, n)) return 1;
    }
    set_err("NVRTC was not found. Set NVRTC_PATH or run the project build script.");
    return 0;
}
#else
static int locate_nvrtc(char* out, size_t n) {
    const char* env = getenv("NVRTC_PATH");
    char exe[1024], parent[1024], cand[1200];
    const char* fixed[] = {"/usr/local/cuda/lib64/libnvrtc.so.12", "/usr/local/cuda/lib64/libnvrtc.so",
                           "libnvrtc.so.12", "libnvrtc.so"};
    size_t i;
    if (env && env[0] && (file_exists(env) || strchr(env, '/') == NULL)) {
        snprintf(out, n, "%s", env);
        return 1;
    }
    exe_dir(exe, sizeof exe);
    parent_of(exe, parent, sizeof parent);
    snprintf(cand, sizeof cand, "%s/.cuda_pkgs/nvidia/cuda_nvrtc/lib/libnvrtc.so.12", parent);
    if (file_exists(cand)) {
        snprintf(out, n, "%s", cand);
        return 1;
    }
    snprintf(cand, sizeof cand, "%s/libnvrtc.so.12", exe);
    if (file_exists(cand)) {
        snprintf(out, n, "%s", cand);
        return 1;
    }
    for (i = 0; i < sizeof fixed / sizeof fixed[0]; ++i) {
        if (fixed[i][0] != '/' || file_exists(fixed[i])) {
            snprintf(out, n, "%s", fixed[i]);
            return 1;
        }
    }
    set_err("NVRTC was not found. Set NVRTC_PATH.");
    return 0;
}
#endif

static int compile_with_nvrtc(const char* source, int want_cubin, char** image, size_t* image_n) {
    char path[1200];
    if (!locate_nvrtc(path, sizeof path)) return 0;
#ifdef _WIN32
    char dirbuf[1200];
    char builtins[1400];
    DWORD load_flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    HMODULE lib;
    snprintf(dirbuf, sizeof dirbuf, "%s", path);
    {
        char* slash = strrchr(dirbuf, '\\');
        char* slash2 = strrchr(dirbuf, '/');
        if (!slash || (slash2 && slash2 > slash)) slash = slash2;
        if (slash) *slash = 0;
    }
    if (find_in_dir(dirbuf, "nvrtc-builtins*.dll", builtins, sizeof builtins))
        LoadLibraryExA(builtins, NULL, load_flags);
    lib = LoadLibraryExA(path, NULL, load_flags);
    if (!lib) {
        set_err("could not load NVRTC from %s", path);
        return 0;
    }
#else
    void* lib = dlopen(path, RTLD_NOW);
    if (!lib) {
        set_err("could not load NVRTC (%s)", path);
        return 0;
    }
#endif
    fn_nvrtcCreateProgram createp = (fn_nvrtcCreateProgram)lib_sym(lib, "nvrtcCreateProgram");
    fn_nvrtcDestroyProgram destroyp = (fn_nvrtcDestroyProgram)lib_sym(lib, "nvrtcDestroyProgram");
    fn_nvrtcCompileProgram compilep = (fn_nvrtcCompileProgram)lib_sym(lib, "nvrtcCompileProgram");
    fn_nvrtcGetPTXSize ptxsize = (fn_nvrtcGetPTXSize)lib_sym(lib, "nvrtcGetPTXSize");
    fn_nvrtcGetPTX getptx = (fn_nvrtcGetPTX)lib_sym(lib, "nvrtcGetPTX");
    fn_nvrtcGetCUBINSize cubsize = (fn_nvrtcGetCUBINSize)lib_sym(lib, "nvrtcGetCUBINSize");
    fn_nvrtcGetCUBIN getcubin = (fn_nvrtcGetCUBIN)lib_sym(lib, "nvrtcGetCUBIN");
    fn_nvrtcGetProgramLogSize logsize = (fn_nvrtcGetProgramLogSize)lib_sym(lib, "nvrtcGetProgramLogSize");
    fn_nvrtcGetProgramLog getlog = (fn_nvrtcGetProgramLog)lib_sym(lib, "nvrtcGetProgramLog");
    if (!createp || !destroyp || !compilep || !ptxsize || !getptx || !logsize || !getlog) {
        set_err("NVRTC is missing compile entry points");
        return 0;
    }
    nvrtcProgram prog;
    if (createp(&prog, source, "mc_risk.cu", 0, NULL, NULL) != 0) {
        set_err("nvrtcCreateProgram failed");
        return 0;
    }
    char arch[64];
    if (want_cubin) snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", g_major, g_minor);
    else snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", g_major, g_minor);
    const char* opts[] = {arch};
    int cerr = compilep(prog, 1, opts);
    if (cerr != 0) {
        size_t lz = 0;
        logsize(prog, &lz);
        char* log = (char*)malloc(lz + 1);
        if (log) {
            getlog(prog, log);
            log[lz] = 0;
            set_err("CUDA compile failed:\n%.1800s", log);
            free(log);
        } else {
            set_err("CUDA compile failed");
        }
        destroyp(&prog);
        return 0;
    }
    size_t bytes = 0;
    if (want_cubin && cubsize && getcubin) {
        if (cubsize(prog, &bytes) != 0 || bytes == 0) {
            destroyp(&prog);
            set_err("nvrtcGetCUBINSize failed");
            return 0;
        }
        char* img = (char*)malloc(bytes);
        if (!img || getcubin(prog, img) != 0) {
            free(img);
            destroyp(&prog);
            set_err("nvrtcGetCUBIN failed");
            return 0;
        }
        *image = img;
        *image_n = bytes;
    } else {
        if (ptxsize(prog, &bytes) != 0 || bytes == 0) {
            destroyp(&prog);
            set_err("nvrtcGetPTXSize failed");
            return 0;
        }
        char* img = (char*)malloc(bytes + 1);
        if (!img || getptx(prog, img) != 0) {
            free(img);
            destroyp(&prog);
            set_err("nvrtcGetPTX failed");
            return 0;
        }
        img[bytes] = 0;
        *image = img;
        *image_n = bytes + 1;
    }
    destroyp(&prog);
    return 1;
}

static int init_cuda(void) {
    if (g_ready) return 1;
    time_init();
    if (!load_driver()) return 0;
    if (!cu_ok(p_cuInit(0), "cuInit")) return 0;
    CUdevice dev = 0;
    if (!cu_ok(p_cuDeviceGet(&dev, 0), "cuDeviceGet")) return 0;
    if (!cu_ok(p_cuDeviceGetName(g_gpu_name, (int)sizeof g_gpu_name, dev), "cuDeviceGetName")) return 0;
    if (!cu_ok(p_cuDeviceGetAttribute(&g_major, 75, dev), "compute major")) return 0;
    if (!cu_ok(p_cuDeviceGetAttribute(&g_minor, 76, dev), "compute minor")) return 0;
    if (!cu_ok(p_cuDeviceGetAttribute(&g_sms, 16, dev), "sms")) return 0;
    if (!cu_ok(p_cuDeviceGetAttribute(&g_clock_khz, 13, dev), "clock")) return 0;
    if (g_sms < 1) g_sms = 1;
    CUcontext ctx = NULL;
    if (!cu_ok(p_cuCtxCreate(&ctx, 0, dev), "cuCtxCreate")) return 0;

    char kernel_path[1200];
    find_kernel(kernel_path, sizeof kernel_path);
    size_t src_n = 0;
    char* source = read_file(kernel_path, &src_n);
    if (!source) {
        set_err("cannot read CUDA source at %s", kernel_path);
        return 0;
    }
    ull hash = fnv64(source, src_n);
    exe_dir(g_cache_dir, sizeof g_cache_dir);
    char cache_cubin[1400], cache_ptx[1400];
    snprintf(cache_cubin, sizeof cache_cubin, "%s%cmc_%d%d_%016llx.cubin", g_cache_dir,
#ifdef _WIN32
             '\\',
#else
             '/',
#endif
             g_major, g_minor, hash);
    snprintf(cache_ptx, sizeof cache_ptx, "%s%cmc_%d%d_%016llx.ptx", g_cache_dir,
#ifdef _WIN32
             '\\',
#else
             '/',
#endif
             g_major, g_minor, hash);

    CUmodule mod = NULL;
    int loaded = 0;
    size_t cached_n = 0;
    char* cached = read_file(cache_cubin, &cached_n);
    if (cached) {
        /* read_file appends a NUL past the cubin. Load the original bytes. */
        if (load_module_bytes(cached, &mod)) loaded = 1;
        free(cached);
    }
    if (!loaded) {
        cached = read_file(cache_ptx, &cached_n);
        if (cached) {
            if (load_module_bytes(cached, &mod)) loaded = 1;
            free(cached);
        }
    }
    if (loaded) {
        g_from_cache = 1;
        g_compile_seconds = 0.0;
        free(source);
    } else {
        double t0 = now_sec();
        char* image = NULL;
        size_t image_n = 0;
        int got = compile_with_nvrtc(source, 1, &image, &image_n);
        int as_cubin = got;
        if (!got) {
            g_err[0] = 0;
            got = compile_with_nvrtc(source, 0, &image, &image_n);
            as_cubin = 0;
        }
        g_compile_seconds = now_sec() - t0;
        free(source);
        if (!got) return 0;
        if (!load_module_bytes(image, &mod)) {
            free(image);
            return 0;
        }
        save_cache(as_cubin ? cache_cubin : cache_ptx, image, image_n);
        free(image);
        g_from_cache = 0;
    }
    if (!bind_functions(mod)) return 0;

    unsigned kat[8];
    CUdeviceptr dkat = 0;
    if (!cu_ok(p_cuMemAlloc(&dkat, sizeof kat), "alloc kat")) return 0;
    void* args[] = {&dkat};
    if (!cu_ok(p_cuLaunchKernel(f_kat, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL), "launch kat")) return 0;
    if (!cu_ok(p_cuCtxSynchronize(), "sync kat")) return 0;
    if (!cu_ok(p_cuMemcpyDtoH(kat, dkat, sizeof kat), "copy kat")) return 0;
    p_cuMemFree(dkat);
    static const unsigned zexp[4] = {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u};
    for (int i = 0; i < 4; ++i) {
        if (kat[i] != zexp[i]) {
            set_err("GPU Philox self-test failed");
            return 0;
        }
    }
    g_ready = 1;
    return 1;
}

static unsigned grid_blocks(void) {
    unsigned b = (unsigned)g_sms * 16u;
    if (b < 64u) b = 64u;
    if (b > 512u) b = 512u;
    return b;
}

static int launch(CUfunction f, unsigned gx, unsigned gy, unsigned shared, void** args, const char* what) {
    if (!cu_ok(p_cuLaunchKernel(f, gx, gy, 1, BLOCK, 1, 1, shared, NULL, args, NULL), what)) return 0;
    if (!cu_ok(p_cuCtxSynchronize(), what)) return 0;
    return 1;
}

/* ---------- CPU pricing baseline ---------- */

typedef struct {
    int tid, nthreads, nopt;
    ull n_paths, seed;
    const double* spots;
    const double* strikes;
    const double* rates;
    const double* divs;
    const double* vols;
    const double* mats;
    const int* is_call;
    double* sum;
    double* sumsq;
} CpuWork;

static void cpu_price_work(CpuWork* w) {
    ull n_pairs = w->n_paths >> 1;
    unsigned k0 = (unsigned)w->seed;
    unsigned k1 = (unsigned)(w->seed >> 32);
    for (int opt = 0; opt < w->nopt; ++opt) {
        double s0 = w->spots[opt];
        double k = w->strikes[opt];
        double r = w->rates[opt];
        double q = w->divs[opt];
        double vol = w->vols[opt];
        double t = w->mats[opt];
        int call = w->is_call[opt];
        double drift = (r - q - 0.5 * vol * vol) * t;
        double scale = vol * sqrt(t);
        double disc = exp(-r * t);
        double local = 0.0, local2 = 0.0;
        for (ull p = (ull)w->tid; p < n_pairs; p += (ull)w->nthreads) {
            unsigned o[4];
            philox4x32_10((unsigned)p, (unsigned)(p >> 32), (unsigned)opt, 1u, k0, k1, o);
            double z, z2;
            box_muller(o[0], o[1], &z, &z2);
            double st_hi = s0 * exp(drift + scale * z);
            double st_lo = s0 * exp(drift - scale * z);
            double pay_hi = call ? fmax(st_hi - k, 0.0) : fmax(k - st_hi, 0.0);
            double pay_lo = call ? fmax(st_lo - k, 0.0) : fmax(k - st_lo, 0.0);
            double avg = 0.5 * disc * (pay_hi + pay_lo);
            local += avg;
            local2 += avg * avg;
        }
        w->sum[opt] = local;
        w->sumsq[opt] = local2;
    }
}

#ifdef _WIN32
static DWORD WINAPI cpu_thread_main(LPVOID p) {
    cpu_price_work((CpuWork*)p);
    return 0;
}
#else
static void* cpu_thread_main(void* p) {
    cpu_price_work((CpuWork*)p);
    return NULL;
}
#endif

static int cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    int n = (int)info.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return n;
}

static int cpu_price(const double* spots, const double* strikes, const double* rates, const double* divs,
                     const double* vols, const double* mats, const int* is_call, int nopt, ull n_paths, ull seed,
                     double* sum, double* sumsq, double* seconds, int* threads_out) {
    int nthreads = cpu_count();
    ull n_pairs = n_paths >> 1;
    if ((ull)nthreads > n_pairs) nthreads = (int)n_pairs;
    if (nthreads < 1) nthreads = 1;
    int stride = nopt < 8 ? 8 : nopt;
    double* acc = (double*)calloc((size_t)nthreads * (size_t)stride, sizeof(double));
    double* acc2 = (double*)calloc((size_t)nthreads * (size_t)stride, sizeof(double));
    CpuWork* works = (CpuWork*)calloc((size_t)nthreads, sizeof(CpuWork));
    if (!acc || !acc2 || !works) {
        free(acc);
        free(acc2);
        free(works);
        set_err("out of memory");
        return 0;
    }
    for (int t = 0; t < nthreads; ++t) {
        works[t].tid = t;
        works[t].nthreads = nthreads;
        works[t].nopt = nopt;
        works[t].n_paths = n_paths;
        works[t].seed = seed;
        works[t].spots = spots;
        works[t].strikes = strikes;
        works[t].rates = rates;
        works[t].divs = divs;
        works[t].vols = vols;
        works[t].mats = mats;
        works[t].is_call = is_call;
        works[t].sum = acc + (size_t)t * (size_t)stride;
        works[t].sumsq = acc2 + (size_t)t * (size_t)stride;
    }
    double t0 = now_sec();
#ifdef _WIN32
    HANDLE* th = (HANDLE*)calloc((size_t)nthreads, sizeof(HANDLE));
    for (int t = 0; t < nthreads; ++t)
        th[t] = CreateThread(NULL, 0, cpu_thread_main, &works[t], 0, NULL);
    for (int t = 0; t < nthreads; ++t)
        if (th[t]) WaitForSingleObject(th[t], INFINITE);
    for (int t = 0; t < nthreads; ++t)
        if (th[t]) CloseHandle(th[t]);
    free(th);
#else
    pthread_t* th = (pthread_t*)calloc((size_t)nthreads, sizeof(pthread_t));
    for (int t = 0; t < nthreads; ++t) pthread_create(&th[t], NULL, cpu_thread_main, &works[t]);
    for (int t = 0; t < nthreads; ++t) pthread_join(th[t], NULL);
    free(th);
#endif
    *seconds = now_sec() - t0;
    for (int opt = 0; opt < nopt; ++opt) {
        sum[opt] = 0.0;
        sumsq[opt] = 0.0;
        for (int t = 0; t < nthreads; ++t) {
            sum[opt] += acc[(size_t)t * (size_t)stride + (size_t)opt];
            sumsq[opt] += acc2[(size_t)t * (size_t)stride + (size_t)opt];
        }
    }
    free(acc);
    free(acc2);
    free(works);
    *threads_out = nthreads;
    return 1;
}

/* ---------- statistics ---------- */

static int hist_quantile(const ull* hist, int nbins, ull under, double lo, double hi, double p, double* q) {
    ull total = under;
    for (int i = 0; i < nbins; ++i) total += hist[i];
    if (total == 0) return 0;
    double target = p * (double)total;
    double width = (hi - lo) / (double)nbins;
    if ((double)under >= target) {
        *q = lo;
        return 1;
    }
    double cdf = (double)under;
    for (int i = 0; i < nbins; ++i) {
        double next = cdf + (double)hist[i];
        if (next >= target) {
            double frac = hist[i] ? (target - cdf) / (double)hist[i] : 0.0;
            *q = lo + ((double)i + frac) * width;
            return 1;
        }
        cdf = next;
    }
    *q = hi;
    return 1;
}

static int hist_es(const ull* hist, int nbins, ull under, double lo, double hi, double p, double* es) {
    ull total = under;
    for (int i = 0; i < nbins; ++i) total += hist[i];
    if (total == 0 || under) return 0;
    double target = p * (double)total;
    double width = (hi - lo) / (double)nbins;
    double cdf = 0.0;
    double tail_sum = 0.0;
    double tail_w = 0.0;
    for (int i = 0; i < nbins; ++i) {
        if (cdf >= target) break;
        double take = (double)hist[i];
        double bin_lo = lo + (double)i * width;
        if (cdf + take <= target) {
            tail_sum += (bin_lo + 0.5 * width) * take;
            tail_w += take;
        } else {
            double need = target - cdf;
            double frac = take > 0.0 ? need / take : 0.0;
            double mid = bin_lo + 0.5 * frac * width;
            tail_sum += mid * need;
            tail_w += need;
            break;
        }
        cdf += take;
    }
    if (tail_w <= 0.0) return 0;
    *es = tail_sum / tail_w;
    return 1;
}

/* ---------- JSON ---------- */

typedef struct {
    char* data;
    size_t len, cap;
} Buf;

static void bprint(Buf* b, const char* fmt, ...) {
    va_list ap;
    for (;;) {
        size_t avail = b->cap - b->len;
        va_start(ap, fmt);
        int n = vsnprintf(b->data + b->len, avail, fmt, ap);
        va_end(ap);
        if (n < 0) return;
        if ((size_t)n < avail) {
            b->len += (size_t)n;
            return;
        }
        size_t cap = (b->len + (size_t)n + 64) * 2;
        char* p = (char*)realloc(b->data, cap);
        if (!p) return;
        b->data = p;
        b->cap = cap;
    }
}

static void bnum(Buf* b, double x) {
    if (!isfinite(x)) bprint(b, "null");
    else bprint(b, "%.10g", x);
}

static void bstr(Buf* b, const char* s) {
    bprint(b, "\"");
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        if (*p == '\\' || *p == '"') bprint(b, "\\%c", *p);
        else if (*p >= 32 && *p < 127) bprint(b, "%c", *p);
        else bprint(b, "\\u%04x", *p);
    }
    bprint(b, "\"");
}

/* ---------- execute ---------- */

typedef struct {
    int ok;
    int n_opt;
    int truncated;
    int cpu_threads;
    double mc, bs, se;
    double var95, var99, es95, es99;
    double speedup;
    double gpu_job_s;
    double geo_mc, geo_closed, geo_se, arith_cv;
    int has_asian;
    char error[1024];
} Summary;

static int gpu_price_options(CUdeviceptr d_spot, CUdeviceptr d_k, CUdeviceptr d_r, CUdeviceptr d_q,
                             CUdeviceptr d_vol, CUdeviceptr d_mat, CUdeviceptr d_call, int nopt, ull n_paths,
                             ull seed, double* sum, double* sumsq, double* seconds) {
    CUdeviceptr d_sum = 0, d_sumsq = 0;
    size_t bytes = (size_t)nopt * sizeof(double);
    if (!cu_ok(p_cuMemAlloc(&d_sum, bytes), "alloc sum")) return 0;
    if (!cu_ok(p_cuMemAlloc(&d_sumsq, bytes), "alloc sumsq")) {
        p_cuMemFree(d_sum);
        return 0;
    }
    p_cuMemsetD8(d_sum, 0, bytes);
    p_cuMemsetD8(d_sumsq, 0, bytes);
    int n_opt = nopt;
    void* args[] = {&d_spot, &d_k, &d_r, &d_q, &d_vol, &d_mat, &d_call, &n_opt, &n_paths, &seed, &d_sum, &d_sumsq};
    double t0 = now_sec();
    int ok = launch(f_price, (unsigned)nopt, grid_blocks(), 0, args, "price_options");
    *seconds = now_sec() - t0;
    if (ok) {
        ok = cu_ok(p_cuMemcpyDtoH(sum, d_sum, bytes), "copy sum") &&
             cu_ok(p_cuMemcpyDtoH(sumsq, d_sumsq, bytes), "copy sumsq");
    }
    p_cuMemFree(d_sum);
    p_cuMemFree(d_sumsq);
    return ok;
}

static double host_book_value(const Job* job) {
    double v = 0.0;
    for (int i = 0; i < job->n_pos; ++i) {
        const Pos* p = &job->pos[i];
        const Und* u = &job->und[p->und];
        if (p->kind == 0) v += p->qty * u->spot;
        else v += p->qty * bs_price(u->spot, p->strike, u->rate, u->div, u->vol, p->mat, p->kind == 1);
    }
    return v;
}

static int execute(const Job* job, const char* json_path, Summary* sum) {
    memset(sum, 0, sizeof *sum);
    if (!init_cuda()) {
        snprintf(sum->error, sizeof sum->error, "%s", g_err);
        return 0;
    }
    double horizon = job->horizon_days / 252.0;
    double L[MAX_U * MAX_U];
    if (!cholesky(job->corr, L, job->n_und)) {
        snprintf(sum->error, sizeof sum->error, "%s", g_err);
        return 0;
    }

    int opt_index[MAX_P];
    int nopt = 0;
    for (int i = 0; i < job->n_pos; ++i)
        if (job->pos[i].kind != 0) opt_index[nopt++] = i;

    double o_spot[MAX_P], o_k[MAX_P], o_r[MAX_P], o_q[MAX_P], o_vol[MAX_P], o_mat[MAX_P];
    int o_call[MAX_P];
    for (int i = 0; i < nopt; ++i) {
        const Pos* p = &job->pos[opt_index[i]];
        const Und* u = &job->und[p->und];
        o_spot[i] = u->spot;
        o_k[i] = p->strike;
        o_r[i] = u->rate;
        o_q[i] = u->div;
        o_vol[i] = u->vol;
        o_mat[i] = p->mat;
        o_call[i] = p->kind == 1;
    }

    double u_spot[MAX_U], u_vol[MAX_U], u_rate[MAX_U], u_div[MAX_U], u_drift[MAX_U];
    int kinds[MAX_P], und_idx[MAX_P];
    double strikes[MAX_P], mats[MAX_P], qtys[MAX_P];
    for (int i = 0; i < job->n_und; ++i) {
        u_spot[i] = job->und[i].spot;
        u_vol[i] = job->und[i].vol;
        u_rate[i] = job->und[i].rate;
        u_div[i] = job->und[i].div;
        u_drift[i] = job->und[i].drift;
    }
    for (int i = 0; i < job->n_pos; ++i) {
        kinds[i] = job->pos[i].kind;
        und_idx[i] = job->pos[i].und;
        strikes[i] = job->pos[i].strike;
        mats[i] = job->pos[i].mat;
        qtys[i] = job->pos[i].qty;
    }

    CUdeviceptr d_spot = 0, d_vol = 0, d_rate = 0, d_div = 0, d_drift = 0, d_L = 0;
    CUdeviceptr d_kinds = 0, d_und = 0, d_strikes = 0, d_mats = 0, d_qtys = 0;
    CUdeviceptr d_os = 0, d_ok = 0, d_or = 0, d_oq = 0, d_ov = 0, d_om = 0, d_oc = 0;
    CUdeviceptr d_book = 0;
    CUdeviceptr keep[32];
    int nkeep = 0;
#define KEEP(p) keep[nkeep++] = (p)
    int failed = 0;
    double device_v0 = 0.0;
    double price_seconds = 0.0, var_seconds = 0.0, asian_seconds = 0.0;
    double bench_gpu_s = 0.0, bench_cpu_s = 0.0;
    int cpu_threads = 0;
    double mc_sum[MAX_P], mc_sumsq[MAX_P];
    double bench_sum[MAX_P], bench_sumsq[MAX_P];
    double cpu_sum[MAX_P], cpu_sumsq[MAX_P];
    memset(mc_sum, 0, sizeof mc_sum);
    memset(mc_sumsq, 0, sizeof mc_sumsq);

    ull* hist = (ull*)calloc(NBINS, sizeof(ull));
    ull under = 0, over = 0;
    double mean = 0, std = 0, q05 = 0, q01 = 0, es05 = 0, es01 = 0;
    int truncated = 0;
    double hist_lo = 0, hist_hi = 0;
    double asian_a = 0, asian_g = 0, asian_a2 = 0, asian_g2 = 0, asian_ag = 0;

    if (!hist) {
        set_err("out of memory");
        failed = 1;
        goto done;
    }

#define UP_D(dst, src, n)                                                              \
    do {                                                                               \
        size_t _b = (size_t)(n) * sizeof(double);                                      \
        if (!cu_ok(p_cuMemAlloc(&(dst), _b), "cuMemAlloc") ||                          \
            !cu_ok(p_cuMemcpyHtoD((dst), (src), _b), "cuMemcpyHtoD")) {                \
            failed = 1;                                                                \
            goto done;                                                                 \
        }                                                                              \
        KEEP(dst);                                                                     \
    } while (0)
#define UP_I(dst, src, n)                                                              \
    do {                                                                               \
        size_t _b = (size_t)(n) * sizeof(int);                                         \
        if (!cu_ok(p_cuMemAlloc(&(dst), _b), "cuMemAlloc") ||                          \
            !cu_ok(p_cuMemcpyHtoD((dst), (src), _b), "cuMemcpyHtoD")) {                \
            failed = 1;                                                                \
            goto done;                                                                 \
        }                                                                              \
        KEEP(dst);                                                                     \
    } while (0)

    UP_D(d_spot, u_spot, job->n_und);
    UP_D(d_vol, u_vol, job->n_und);
    UP_D(d_rate, u_rate, job->n_und);
    UP_D(d_div, u_div, job->n_und);
    UP_D(d_drift, u_drift, job->n_und);
    UP_D(d_L, L, job->n_und * job->n_und);
    UP_I(d_kinds, kinds, job->n_pos);
    UP_I(d_und, und_idx, job->n_pos);
    UP_D(d_strikes, strikes, job->n_pos);
    UP_D(d_mats, mats, job->n_pos);
    UP_D(d_qtys, qtys, job->n_pos);
    if (nopt > 0) {
        UP_D(d_os, o_spot, nopt);
        UP_D(d_ok, o_k, nopt);
        UP_D(d_or, o_r, nopt);
        UP_D(d_oq, o_q, nopt);
        UP_D(d_ov, o_vol, nopt);
        UP_D(d_om, o_mat, nopt);
        UP_I(d_oc, o_call, nopt);
        if (!gpu_price_options(d_os, d_ok, d_or, d_oq, d_ov, d_om, d_oc, nopt, job->paths, job->seed, mc_sum,
                               mc_sumsq, &price_seconds)) {
            failed = 1;
            goto done;
        }
        if (job->bench_paths == job->paths) {
            bench_gpu_s = price_seconds;
            memcpy(bench_sum, mc_sum, sizeof bench_sum);
            memcpy(bench_sumsq, mc_sumsq, sizeof bench_sumsq);
        } else if (!gpu_price_options(d_os, d_ok, d_or, d_oq, d_ov, d_om, d_oc, nopt, job->bench_paths, job->seed,
                                      bench_sum, bench_sumsq, &bench_gpu_s)) {
            failed = 1;
            goto done;
        }
        if (!cpu_price(o_spot, o_k, o_r, o_q, o_vol, o_mat, o_call, nopt, job->bench_paths, job->seed, cpu_sum,
                       cpu_sumsq, &bench_cpu_s, &cpu_threads)) {
            failed = 1;
            goto done;
        }
    }

    if (!cu_ok(p_cuMemAlloc(&d_book, sizeof(double)), "alloc book")) {
        failed = 1;
        goto done;
    }
    KEEP(d_book);
    {
        int n_pos = job->n_pos;
        void* args[] = {&d_spot, &d_rate, &d_div, &d_vol, &d_kinds, &d_und, &d_strikes, &d_mats, &d_qtys, &n_pos,
                        &d_book};
        if (!cu_ok(p_cuLaunchKernel(f_book, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL), "book_value") ||
            !cu_ok(p_cuCtxSynchronize(), "book sync") ||
            !cu_ok(p_cuMemcpyDtoH(&device_v0, d_book, sizeof device_v0), "book copy")) {
            failed = 1;
            goto done;
        }
    }

    {
        CUdeviceptr d_sum = 0, d_sumsq = 0;
        if (!cu_ok(p_cuMemAlloc(&d_sum, sizeof(double)), "alloc pnl") ||
            !cu_ok(p_cuMemAlloc(&d_sumsq, sizeof(double)), "alloc pnl2")) {
            if (d_sum) p_cuMemFree(d_sum);
            failed = 1;
            goto done;
        }
        p_cuMemsetD8(d_sum, 0, sizeof(double));
        p_cuMemsetD8(d_sumsq, 0, sizeof(double));
        int n_und = job->n_und, n_pos = job->n_pos;
        ull n_paths = job->paths;
        ull seed = job->seed ^ 0x9E3779B97F4A7C15ull;
        void* args[] = {&d_spot, &d_vol, &d_rate, &d_div, &d_drift, &d_L, &n_und, &d_kinds, &d_und,
                        &d_strikes, &d_mats, &d_qtys, &n_pos, &horizon, &n_paths, &seed, &d_sum, &d_sumsq};
        double t0 = now_sec();
        int ok = launch(f_moments, grid_blocks(), 1, 0, args, "var_moments");
        var_seconds = now_sec() - t0;
        double sum_pnl = 0, sum_pnl_sq = 0;
        if (!ok || !cu_ok(p_cuMemcpyDtoH(&sum_pnl, d_sum, sizeof sum_pnl), "pnl") ||
            !cu_ok(p_cuMemcpyDtoH(&sum_pnl_sq, d_sumsq, sizeof sum_pnl_sq), "pnl2")) {
            p_cuMemFree(d_sum);
            p_cuMemFree(d_sumsq);
            failed = 1;
            goto done;
        }
        p_cuMemFree(d_sum);
        p_cuMemFree(d_sumsq);
        mean = sum_pnl / (double)n_paths;
        double var = sum_pnl_sq / (double)n_paths - mean * mean;
        if (var < 0.0) var = 0.0;
        std = sqrt(var);

        if (std < 1e-8) {
            q05 = q01 = es05 = es01 = mean;
            hist_lo = mean - 1.0;
            hist_hi = mean + 1.0;
        } else {
            CUdeviceptr d_hist = 0, d_under = 0, d_over = 0;
            if (!cu_ok(p_cuMemAlloc(&d_hist, NBINS * sizeof(ull)), "alloc hist") ||
                !cu_ok(p_cuMemAlloc(&d_under, sizeof(ull)), "alloc under") ||
                !cu_ok(p_cuMemAlloc(&d_over, sizeof(ull)), "alloc over")) {
                if (d_hist) p_cuMemFree(d_hist);
                if (d_under) p_cuMemFree(d_under);
                failed = 1;
                goto done;
            }
            double span = 12.0;
            int covered = 0;
            for (int attempt = 0; attempt < 3 && !covered; ++attempt) {
                hist_lo = mean - span * std;
                hist_hi = mean + span * std;
                p_cuMemsetD8(d_hist, 0, NBINS * sizeof(ull));
                p_cuMemsetD8(d_under, 0, sizeof(ull));
                p_cuMemsetD8(d_over, 0, sizeof(ull));
                int nbins = NBINS;
                void* hargs[] = {&d_spot, &d_vol,   &d_rate, &d_div,    &d_drift, &d_L,     &n_und,  &d_kinds,
                                 &d_und,  &d_strikes, &d_mats, &d_qtys,   &n_pos,   &horizon, &n_paths, &seed,
                                 &hist_lo, &hist_hi, &nbins, &d_hist,   &d_under, &d_over};
                double t1 = now_sec();
                ok = launch(f_hist, grid_blocks(), 1, NBINS * (unsigned)sizeof(unsigned int), hargs, "var_hist");
                var_seconds += now_sec() - t1;
                under = 0;
                over = 0;
                if (!ok || !cu_ok(p_cuMemcpyDtoH(hist, d_hist, NBINS * sizeof(ull)), "hist") ||
                    !cu_ok(p_cuMemcpyDtoH(&under, d_under, sizeof under), "under") ||
                    !cu_ok(p_cuMemcpyDtoH(&over, d_over, sizeof over), "over")) {
                    p_cuMemFree(d_hist);
                    p_cuMemFree(d_under);
                    p_cuMemFree(d_over);
                    failed = 1;
                    goto done;
                }
                ull tail = under + over;
                if (tail == 0 || tail * 1000ull < n_paths) covered = 1;
                else span *= 2.0;
            }
            truncated = !covered;
            p_cuMemFree(d_hist);
            p_cuMemFree(d_under);
            p_cuMemFree(d_over);
            if (!hist_quantile(hist, NBINS, under, hist_lo, hist_hi, 0.05, &q05) ||
                !hist_quantile(hist, NBINS, under, hist_lo, hist_hi, 0.01, &q01)) {
                set_err("histogram was empty");
                failed = 1;
                goto done;
            }
            if (!hist_es(hist, NBINS, under, hist_lo, hist_hi, 0.05, &es05)) es05 = q05;
            if (!hist_es(hist, NBINS, under, hist_lo, hist_hi, 0.01, &es01)) es01 = q01;
        }
    }

    if (job->has_asian) {
        const Und* u = &job->und[job->asian_und];
        CUdeviceptr ds[5] = {0, 0, 0, 0, 0};
        for (int k = 0; k < 5; ++k) {
            if (!cu_ok(p_cuMemAlloc(&ds[k], sizeof(double)), "alloc asian")) {
                for (int j = 0; j < k; ++j) p_cuMemFree(ds[j]);
                failed = 1;
                goto done;
            }
            p_cuMemsetD8(ds[k], 0, sizeof(double));
        }
        double spot = u->spot, strike = job->asian_k, rate = u->rate, divd = u->div, vol = u->vol, mat = job->asian_mat;
        int steps = job->asian_steps;
        ull n_paths = job->paths;
        ull seed = job->seed ^ 0xD1B54A32D192ED03ull;
        void* args[] = {&spot, &strike, &rate, &divd, &vol, &mat, &steps, &n_paths, &seed,
                        &ds[0], &ds[1], &ds[2], &ds[3], &ds[4]};
        double t0 = now_sec();
        int ok = launch(f_asian, grid_blocks(), 1, 0, args, "asian_mc");
        asian_seconds = now_sec() - t0;
        double outs[5];
        for (int k = 0; k < 5 && ok; ++k)
            ok = cu_ok(p_cuMemcpyDtoH(&outs[k], ds[k], sizeof(double)), "asian copy");
        for (int k = 0; k < 5; ++k) p_cuMemFree(ds[k]);
        if (!ok) {
            failed = 1;
            goto done;
        }
        asian_a = outs[0];
        asian_g = outs[1];
        asian_a2 = outs[2];
        asian_g2 = outs[3];
        asian_ag = outs[4];
    }

done:
    for (int i = 0; i < nkeep; ++i)
        if (keep[i]) p_cuMemFree(keep[i]);
    if (failed) {
        free(hist);
        snprintf(sum->error, sizeof sum->error, "%s", g_err);
        return 0;
    }

    double host_v0 = host_book_value(job);
    ull n_pairs = job->paths >> 1;
    double speedup = (nopt > 0 && bench_gpu_s > 0.0) ? bench_cpu_s / bench_gpu_s : 0.0;
    int cores = 128;
    if (g_major == 7) cores = 64;
    if (g_major == 8 && g_minor == 0) cores = 64;
    if (g_major >= 9 && g_minor == 0) cores = 128;
    double tflops = (double)g_sms * (double)cores * 2.0 * ((double)g_clock_khz * 1000.0) / 1e12;

    double delta_u[MAX_U], vega_u[MAX_U];
    memset(delta_u, 0, sizeof delta_u);
    memset(vega_u, 0, sizeof vega_u);

    Buf buf = {0};
    buf.data = (char*)malloc(256);
    buf.cap = 256;
    buf.data[0] = 0;
    bprint(&buf, "{\n");
    bprint(&buf, "  \"engine\": \"mc_risk\",\n");
    bprint(&buf, "  \"gpu\": {\"name\": ");
    bstr(&buf, g_gpu_name);
    bprint(&buf, ", \"compute_capability\": \"%d.%d\", \"sms\": %d, \"clock_mhz\": ", g_major, g_minor, g_sms);
    bnum(&buf, (double)g_clock_khz / 1000.0);
    bprint(&buf, ", \"fp32_tflops\": ");
    bnum(&buf, tflops);
    bprint(&buf, ", \"cubin_cached\": %s},\n", g_from_cache ? "true" : "false");
    bprint(&buf, "  \"paths\": %llu,\n", job->paths);
    bprint(&buf, "  \"bench_paths\": %llu,\n", job->bench_paths);
    bprint(&buf, "  \"seed\": %llu,\n", job->seed);
    bprint(&buf, "  \"horizon_days\": ");
    bnum(&buf, job->horizon_days);
    bprint(&buf, ",\n  \"horizon_years\": ");
    bnum(&buf, horizon);
    bprint(&buf, ",\n  \"portfolio_value\": ");
    bnum(&buf, host_v0);
    bprint(&buf, ",\n  \"device_portfolio_value\": ");
    bnum(&buf, device_v0);
    bprint(&buf, ",\n  \"model\": \"Constant-volatility geometric Brownian motion. ");
    bprint(&buf, "Option prices use antithetic Monte Carlo and are checked against Black-Scholes. ");
    bprint(&buf, "Portfolio scenarios are correlated with a Cholesky factor, then the book is revalued. ");
    bprint(&buf, "The horizon is measured in trading days out of 252. ");
    bprint(&buf, "Drift is the price drift; the sample book uses rate minus dividend.\",\n");

    bprint(&buf, "  \"timing\": {\n");
    bprint(&buf, "    \"compile_seconds\": ");
    bnum(&buf, g_compile_seconds);
    bprint(&buf, ",\n    \"gpu_price_seconds\": ");
    bnum(&buf, price_seconds);
    bprint(&buf, ",\n    \"gpu_var_seconds\": ");
    bnum(&buf, var_seconds);
    bprint(&buf, ",\n    \"gpu_asian_seconds\": ");
    bnum(&buf, asian_seconds);
    double job_s = price_seconds + var_seconds + asian_seconds;
    bprint(&buf, ",\n    \"gpu_job_seconds\": ");
    bnum(&buf, job_s);
    bprint(&buf, ",\n    \"bench_gpu_seconds\": ");
    bnum(&buf, bench_gpu_s);
    bprint(&buf, ",\n    \"bench_cpu_seconds\": ");
    bnum(&buf, bench_cpu_s);
    bprint(&buf, ",\n    \"cpu_threads\": %d,\n", cpu_threads);
    bprint(&buf, "    \"speedup\": ");
    bnum(&buf, speedup);
    bprint(&buf, ",\n    \"gpu_price_paths_per_sec\": ");
    bnum(&buf, bench_gpu_s > 0.0 ? (double)job->bench_paths / bench_gpu_s : 0.0);
    bprint(&buf, ",\n    \"cpu_price_paths_per_sec\": ");
    bnum(&buf, bench_cpu_s > 0.0 ? (double)job->bench_paths / bench_cpu_s : 0.0);
    bprint(&buf, ",\n    \"var_paths_per_sec\": ");
    bnum(&buf, var_seconds > 0.0 ? (double)job->paths / var_seconds : 0.0);
    bprint(&buf, "\n  },\n");

    bprint(&buf, "  \"book\": [\n");
    for (int i = 0; i < job->n_pos; ++i) {
        const Pos* p = &job->pos[i];
        const Und* u = &job->und[p->und];
        double unit = 0, delta = 0, vega = 0, mc = 0, se = 0;
        int has_mc = 0;
        if (p->kind == 0) {
            unit = u->spot;
            delta = 1.0;
            vega = 0.0;
        } else {
            bs_greeks(u->spot, p->strike, u->rate, u->div, u->vol, p->mat, p->kind == 1, &unit, &delta, &vega);
            int oi = -1;
            for (int k = 0; k < nopt; ++k)
                if (opt_index[k] == i) oi = k;
            if (oi >= 0 && n_pairs > 0) {
                has_mc = 1;
                mc = mc_sum[oi] / (double)n_pairs;
                double second = mc_sumsq[oi] / (double)n_pairs;
                double vv = second - mc * mc;
                if (vv < 0.0) vv = 0.0;
                se = sqrt(vv / (double)n_pairs);
            }
        }
        delta_u[p->und] += p->qty * delta;
        vega_u[p->und] += p->qty * vega;
        bprint(&buf, "    {\"name\": ");
        bstr(&buf, u->name);
        bprint(&buf, ", \"kind\": \"%s\", \"quantity\": ", p->kind == 0 ? "equity" : p->kind == 1 ? "call" : "put");
        bnum(&buf, p->qty);
        bprint(&buf, ", \"strike\": ");
        bnum(&buf, p->strike);
        bprint(&buf, ", \"maturity_years\": ");
        bnum(&buf, p->mat);
        bprint(&buf, ", \"unit_value\": ");
        bnum(&buf, unit);
        bprint(&buf, ", \"delta\": ");
        bnum(&buf, delta);
        bprint(&buf, ", \"vega\": ");
        bnum(&buf, vega);
        bprint(&buf, ", \"line_value\": ");
        bnum(&buf, unit * p->qty);
        if (has_mc) {
            bprint(&buf, ", \"monte_carlo\": ");
            bnum(&buf, mc);
            bprint(&buf, ", \"std_error\": ");
            bnum(&buf, se);
            bprint(&buf, ", \"ci95\": [");
            bnum(&buf, mc - 1.96 * se);
            bprint(&buf, ", ");
            bnum(&buf, mc + 1.96 * se);
            bprint(&buf, "], \"rel_error\": ");
            bnum(&buf, unit != 0.0 ? (mc - unit) / unit : 0.0);
            int inside = mc - 1.96 * se <= unit && unit <= mc + 1.96 * se;
            bprint(&buf, ", \"black_scholes_inside_ci\": %s", inside ? "true" : "false");
            if (i == opt_index[0]) {
                sum->mc = mc;
                sum->bs = unit;
                sum->se = se;
            }
        } else {
            bprint(&buf, ", \"monte_carlo\": null, \"std_error\": null");
        }
        bprint(&buf, "}%s\n", i + 1 == job->n_pos ? "" : ",");
    }
    bprint(&buf, "  ],\n");

    bprint(&buf, "  \"underlyings\": [\n");
    for (int i = 0; i < job->n_und; ++i) {
        bprint(&buf, "    {\"name\": ");
        bstr(&buf, job->und[i].name);
        bprint(&buf, ", \"delta\": ");
        bnum(&buf, delta_u[i]);
        bprint(&buf, ", \"vega\": ");
        bnum(&buf, vega_u[i]);
        bprint(&buf, "}%s\n", i + 1 == job->n_und ? "" : ",");
    }
    bprint(&buf, "  ],\n");

    if (job->has_asian) {
        const Und* u = &job->und[job->asian_und];
        double n = (double)job->paths;
        double mean_a = asian_a / n;
        double mean_g = asian_g / n;
        double var_g = asian_g2 / n - mean_g * mean_g;
        double var_a = asian_a2 / n - mean_a * mean_a;
        if (var_g < 0.0) var_g = 0.0;
        if (var_a < 0.0) var_a = 0.0;
        double cov = asian_ag / n - mean_a * mean_g;
        double beta = var_g > 1e-18 ? cov / var_g : 0.0;
        double closed = geo_asian_call(u->spot, job->asian_k, u->rate, u->div, u->vol, job->asian_mat, job->asian_steps);
        double cv = mean_a - beta * (mean_g - closed);
        double se_g = sqrt(var_g / n);
        double var_cv = var_a + beta * beta * var_g - 2.0 * beta * cov;
        if (var_cv < 0.0) var_cv = 0.0;
        double se_cv = sqrt(var_cv / n);
        sum->has_asian = 1;
        sum->geo_mc = mean_g;
        sum->geo_closed = closed;
        sum->geo_se = se_g;
        sum->arith_cv = cv;
        bprint(&buf, "  \"asian\": {\"name\": ");
        bstr(&buf, u->name);
        bprint(&buf, ", \"strike\": ");
        bnum(&buf, job->asian_k);
        bprint(&buf, ", \"maturity_years\": ");
        bnum(&buf, job->asian_mat);
        bprint(&buf, ", \"steps\": %d, \"quantity\": ", job->asian_steps);
        bnum(&buf, job->asian_qty);
        bprint(&buf, ", \"arithmetic_monte_carlo\": ");
        bnum(&buf, mean_a);
        bprint(&buf, ", \"geometric_monte_carlo\": ");
        bnum(&buf, mean_g);
        bprint(&buf, ", \"geometric_closed_form\": ");
        bnum(&buf, closed);
        bprint(&buf, ", \"geometric_std_error\": ");
        bnum(&buf, se_g);
        bprint(&buf, ", \"control_variate_price\": ");
        bnum(&buf, cv);
        bprint(&buf, ", \"control_variate_std_error\": ");
        bnum(&buf, se_cv);
        bprint(&buf, ", \"beta\": ");
        bnum(&buf, beta);
        bprint(&buf, ", \"line_value\": ");
        bnum(&buf, cv * job->asian_qty);
        bprint(&buf, "},\n");
    } else {
        bprint(&buf, "  \"asian\": null,\n");
    }

    bprint(&buf, "  \"var\": {\n");
    bprint(&buf, "    \"mean_pnl\": ");
    bnum(&buf, mean);
    bprint(&buf, ",\n    \"std_pnl\": ");
    bnum(&buf, std);
    bprint(&buf, ",\n    \"pnl_quantile_05\": ");
    bnum(&buf, q05);
    bprint(&buf, ",\n    \"pnl_quantile_01\": ");
    bnum(&buf, q01);
    bprint(&buf, ",\n    \"var_95\": ");
    bnum(&buf, -q05);
    bprint(&buf, ",\n    \"var_99\": ");
    bnum(&buf, -q01);
    bprint(&buf, ",\n    \"es_95\": ");
    bnum(&buf, -es05);
    bprint(&buf, ",\n    \"es_99\": ");
    bnum(&buf, -es01);
    bprint(&buf, ",\n    \"histogram_truncated\": %s,\n", truncated ? "true" : "false");
    bprint(&buf, "    \"hist_lo\": ");
    bnum(&buf, hist_lo);
    bprint(&buf, ",\n    \"hist_hi\": ");
    bnum(&buf, hist_hi);
    bprint(&buf, ",\n    \"scenarios\": %llu,\n", job->paths);
    bprint(&buf, "    \"hist_bins\": [");
    int factor = NBINS / DISP_BINS;
    for (int i = 0; i < DISP_BINS; ++i) {
        ull c = 0;
        for (int k = 0; k < factor; ++k) c += hist[i * factor + k];
        bprint(&buf, "%s%llu", i ? "," : "", c);
    }
    bprint(&buf, "]\n  }\n}\n");

    if (json_path) {
        FILE* out = fopen(json_path, "wb");
        if (!out) {
            free(hist);
            free(buf.data);
            snprintf(sum->error, sizeof sum->error, "cannot write %s", json_path);
            return 0;
        }
        fwrite(buf.data, 1, buf.len, out);
        fclose(out);
    }
    free(buf.data);
    free(hist);

    sum->ok = 1;
    sum->n_opt = nopt;
    sum->truncated = truncated;
    sum->cpu_threads = cpu_threads;
    sum->var95 = -q05;
    sum->var99 = -q01;
    sum->es95 = -es05;
    sum->es99 = -es01;
    sum->speedup = speedup;
    sum->gpu_job_s = job_s;
    fprintf(stderr,
            "%s  value %.4f  VaR95 %.4f  VaR99 %.4f  ES99 %.4f  GPU job %.3fs  speedup %.2fx (%d CPU threads)\n",
            g_gpu_name, host_v0, sum->var95, sum->var99, sum->es99, job_s, speedup, cpu_threads);
    return 1;
}

static int expect_close(const char* name, double got, double want, double tol) {
    double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
    if (fabs(got - want) > tol * scale) {
        fprintf(stderr, "FAIL %s got %.8g want %.8g\n", name, got, want);
        return 0;
    }
    fprintf(stderr, "PASS %s  %.8g\n", name, got);
    return 1;
}

static int selftest(void) {
    int pass = 1;
    time_init();
    if (!kat_cpu()) {
        fprintf(stderr, "FAIL CPU Philox test vector\n");
        return 1;
    }
    fprintf(stderr, "PASS CPU Philox4x32-10 test vectors\n");
    double call = bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, 1);
    double put = bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, 0);
    pass &= expect_close("Black-Scholes ATM call", call, 10.450583572186, 1e-9);
    double parity = call - put - (100.0 - 100.0 * exp(-0.05));
    pass &= expect_close("put-call parity gap", parity, 0.0, 1e-9);

    if (!init_cuda()) {
        fprintf(stderr, "FAIL CUDA init: %s\n", g_err);
        return 1;
    }
    fprintf(stderr, "PASS GPU Philox on %s (sm_%d%d, %d SMs)\n", g_gpu_name, g_major, g_minor, g_sms);
    {
        double host = call;
        double device = 0.0;
        CUdeviceptr d = 0;
        if (!cu_ok(p_cuMemAlloc(&d, sizeof(double)), "bs alloc")) {
            fprintf(stderr, "FAIL %s\n", g_err);
            return 1;
        }
        double s = 100, k = 100, r = 0.05, q = 0, vol = 0.2, t = 1;
        int is_call = 1;
        void* args[] = {&s, &k, &r, &q, &vol, &t, &is_call, &d};
        if (!cu_ok(p_cuLaunchKernel(f_bs, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL), "bs launch") ||
            !cu_ok(p_cuCtxSynchronize(), "bs sync") || !cu_ok(p_cuMemcpyDtoH(&device, d, sizeof device), "bs copy")) {
            fprintf(stderr, "FAIL device Black-Scholes: %s\n", g_err);
            p_cuMemFree(d);
            return 1;
        }
        p_cuMemFree(d);
        pass &= expect_close("device Black-Scholes vs host", device, host, 1e-8);
    }

    char equity_txt[] =
        "MCJOB 1\n"
        "seed 7\n"
        "paths 4000000\n"
        "bench_paths 200000\n"
        "horizon_days 1\n"
        "underlyings 1\n"
        "EQ 100 0.2 0.05 0 0.05\n"
        "correlation\n"
        "1\n"
        "positions 1\n"
        "0 equity 0 0 1\n";
    Job equity;
    if (!parse_job(equity_txt, &equity)) {
        fprintf(stderr, "FAIL parse equity job: %s\n", g_err);
        return 1;
    }
    Summary es;
    if (!execute(&equity, NULL, &es)) {
        fprintf(stderr, "FAIL equity run: %s\n", es.error);
        return 1;
    }
    double dt = 1.0 / 252.0;
    double sig = 0.2, mu = 0.05, S = 100.0;
    double zp = norm_ppf(0.01);
    double m = (mu - 0.5 * sig * sig) * dt;
    double s = sig * sqrt(dt);
    double analytic_var99 = -(S * exp(m + s * zp) - S);
    double analytic_es = S - S * exp(mu * dt) * norm_cdf(zp - s) / 0.01;
    pass &= expect_close("1-day 99% VaR vs lognormal", es.var99, analytic_var99, 0.02);
    pass &= expect_close("1-day 99% ES vs lognormal", es.es99, analytic_es, 0.04);
    if (es.truncated) {
        fprintf(stderr, "FAIL equity histogram was truncated\n");
        pass = 0;
    } else {
        fprintf(stderr, "PASS equity histogram covers the tails\n");
    }

    char opt_txt[] =
        "MCJOB 1\n"
        "seed 11\n"
        "paths 2000000\n"
        "bench_paths 1000000\n"
        "horizon_days 10\n"
        "underlyings 1\n"
        "EQ 100 0.2 0.05 0 0.05\n"
        "correlation\n"
        "1\n"
        "positions 1\n"
        "0 call 100 1 1\n"
        "asian 0 100 1 16 1\n";
    Job opt;
    if (!parse_job(opt_txt, &opt)) {
        fprintf(stderr, "FAIL parse option job: %s\n", g_err);
        return 1;
    }
    Summary os;
    if (!execute(&opt, NULL, &os)) {
        fprintf(stderr, "FAIL option run: %s\n", os.error);
        return 1;
    }
    if (fabs(os.mc - os.bs) > fmax(4.0 * os.se, 0.02)) {
        fprintf(stderr, "FAIL MC %.8g BS %.8g se %.8g\n", os.mc, os.bs, os.se);
        pass = 0;
    } else {
        fprintf(stderr, "PASS Monte Carlo call %.6f vs Black-Scholes %.6f (se %.6f)\n", os.mc, os.bs, os.se);
    }
    if (!os.has_asian || fabs(os.geo_mc - os.geo_closed) > fmax(5.0 * os.geo_se, 0.02)) {
        fprintf(stderr, "FAIL geometric Asian MC %.8g closed %.8g se %.8g\n", os.geo_mc, os.geo_closed, os.geo_se);
        pass = 0;
    } else {
        fprintf(stderr, "PASS geometric Asian %.6f vs closed form %.6f\n", os.geo_mc, os.geo_closed);
    }
    if (!(os.arith_cv > os.geo_closed - 0.05)) {
        fprintf(stderr, "FAIL arithmetic Asian control variate %.8g below geometric %.8g\n", os.arith_cv, os.geo_closed);
        pass = 0;
    } else {
        fprintf(stderr, "PASS arithmetic Asian control variate %.6f\n", os.arith_cv);
    }
    if (!(os.speedup > 1.5)) {
        fprintf(stderr, "FAIL speedup %.3f is not above 1.5\n", os.speedup);
        pass = 0;
    } else {
        fprintf(stderr, "PASS GPU/CPU speedup %.2fx on %llu paths (%d threads)\n", os.speedup, opt.bench_paths,
                os.cpu_threads);
    }

    if (!pass) {
        fprintf(stderr, "SELFTEST FAILED\n");
        return 1;
    }
    fprintf(stderr, "SELFTEST PASSED\n");
    return 0;
}

int main(int argc, char** argv) {
    time_init();
    if (argc == 2 && strcmp(argv[1], "selftest") == 0) return selftest();
    if (argc == 4 && strcmp(argv[1], "run") == 0) {
        Job job;
        if (!parse_file(argv[2], &job)) {
            fprintf(stderr, "%s\n", g_err);
            return 1;
        }
        Summary sum;
        if (!execute(&job, argv[3], &sum)) {
            fprintf(stderr, "%s\n", sum.error[0] ? sum.error : g_err);
            return 1;
        }
        return 0;
    }
    fprintf(stderr, "usage:\n  mc_engine selftest\n  mc_engine run job.txt result.json\n");
    return 2;
}
