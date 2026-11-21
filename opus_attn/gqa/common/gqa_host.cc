// Host harness of the GQA forward kernels: varlen inputs, the opus kernel of the arch and aiter's asm code object (where
// there is one) on the same inputs, a CPU reference, A/B comparison and timing. The traits of the opus kernel give its
// kernel arguments, grid and the problems it supports.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(GQA_ARCH_GFX950)
#include "gfx950/gqa_traits.h"
template <class Traits>
__global__ void gqa_d128_kernel(opus_gqa_kargs kargs);
template <class Traits>
__global__ void gqa_d512_kernel(opus_gqa_kargs kargs);
#elif defined(GQA_ARCH_GFX1250)
#include "gfx1250/gqa_traits.h"
template <class Traits>
__global__ void gqa_d128_kernel(opus_gqa_varlen_kargs kargs);
#else
#error "No target arch defined. The Makefile passes GQA_ARCH_<ARCH> from ARCH (e.g. ARCH=gfx950)."
#endif

#define CHECK_HIP(x)                                                                                \
    do {                                                                                            \
        hipError_t e_ = (x);                                                                        \
        if (e_ != hipSuccess) {                                                                     \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

namespace {

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
float bf2f(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
float uniform(uint64_t seed, uint64_t i) {  // [-1, 1)
    return float(splitmix(seed * 0x100000001b3ull + i) >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

template <class F>
void parallel_for(int64_t n, F f) {
    int nt = std::max(1u, std::thread::hardware_concurrency());
    nt = int(std::min<int64_t>(nt, std::max<int64_t>(1, n)));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for (int64_t i = t; i < n; i += nt) f(i);
        });
    for (auto& x : th) x.join();
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

std::string aiter_dir() {
    if (const char* a = std::getenv("AITER_DIR")) return a;
    const char* h = std::getenv("HOME");
    return std::string(h ? h : ".") + "/develop/aiter";
}

// Q/O [total_q, H, D], K/V [total_k, H_KV, D], bf16; batch b has sq[b] Q rows and sk[b] KV rows.
struct Problem {
    std::vector<int> sq, sk;
    int hq = 64, hkv = 8, d = 128;
    bool causal = true;
    bool sink = false;
    float scale = 0.0f;
    float qscale = 1.0f;  // input Q scale: large values exercise the row-max rescale
    uint64_t seed = 1;

    int batch() const { return int(sq.size()); }
    int64_t total_q() const { int64_t t = 0; for (int x : sq) t += x; return t; }
    int64_t total_k() const { int64_t t = 0; for (int x : sk) t += x; return t; }
    int max_q() const { return *std::max_element(sq.begin(), sq.end()); }
    bool fixed_len() const {  // one length for every batch, sq == sk
        for (int b = 0; b < batch(); ++b)
            if (sq[b] != sq[0] || sk[b] != sq[0]) return false;
        return true;
    }
    // Keys row i of batch b sees: [0, kv_end(b, i)).
    int kv_end(int b, int i) const {
        if (!causal) return sk[b];
        return std::clamp(i + (sk[b] - sq[b]) + 1, 0, sk[b]);
    }
    double pairs() const {
        double p = 0;
        for (int b = 0; b < batch(); ++b)
            for (int i = 0; i < sq[b]; ++i) p += kv_end(b, i);
        return p;
    }
};

struct Options {
    std::string kernels;
    std::string asm_dir;  // empty: from aiter_dir()
    int opt = 0x7;        // OPUS_GQA_OPT_* bits of the varlen kernel arguments
    bool verify = false, perf = false;
    int iters = 20, warmup = 5;
    double ref_budget = 4e9;
};

struct HostData {
    std::vector<uint16_t> q, k, v;
    std::vector<int> cu_q, cu_k;
    std::vector<float> sink;
};

HostData make_data(const Problem& p) {
    HostData h;
    const int64_t tq = p.total_q(), tk = p.total_k();
    h.q.resize(tq * p.hq * p.d);
    h.k.resize(tk * p.hkv * p.d);
    h.v.resize(tk * p.hkv * p.d);
    parallel_for(int64_t(h.q.size() >> 12) + 1, [&](int64_t c) {
        for (int64_t i = c << 12; i < std::min<int64_t>(h.q.size(), (c + 1) << 12); ++i)
            h.q[i] = f2bf(p.qscale * uniform(p.seed, i));
    });
    parallel_for(int64_t(h.k.size() >> 12) + 1, [&](int64_t c) {
        for (int64_t i = c << 12; i < std::min<int64_t>(h.k.size(), (c + 1) << 12); ++i) {
            h.k[i] = f2bf(uniform(p.seed + 1, i));
            h.v[i] = f2bf(uniform(p.seed + 2, i));
        }
    });
    h.cu_q.assign(1, 0);
    h.cu_k.assign(1, 0);
    for (int b = 0; b < p.batch(); ++b) {
        h.cu_q.push_back(h.cu_q.back() + p.sq[b]);
        h.cu_k.push_back(h.cu_k.back() + p.sk[b]);
    }
    h.sink.resize(p.hq);
    for (int i = 0; i < p.hq; ++i) h.sink[i] = 2.0f * uniform(p.seed + 3, i);  // scaled-logit domain
    return h;
}

struct RefRow {
    int b, i, h;
    std::vector<float> o;
    float lse;
};

// Reference rows: every row for small problems, an even sample (always including each batch's first and last rows)
// otherwise.
std::vector<RefRow> cpu_reference(const Problem& p, const HostData& d, double budget) {
    std::vector<std::tuple<int, int, int>> rows;
    const double cost = p.pairs() * p.hq * p.d / 128.0;
    const int64_t total_rows = p.total_q() * p.hq;
    const int64_t want = cost <= budget ? total_rows : std::max<int64_t>(64, int64_t(total_rows * budget / cost));
    const int64_t step = std::max<int64_t>(1, total_rows / want);
    for (int b = 0; b < p.batch(); ++b)
        for (int h = 0; h < p.hq; ++h) {
            rows.emplace_back(b, 0, h);
            if (p.sq[b] > 1) rows.emplace_back(b, p.sq[b] - 1, h);
        }
    for (int64_t r = 0; r < total_rows; r += step) {
        int64_t t = r / p.hq;
        int h = int(r % p.hq), b = 0;
        while (t >= d.cu_q[b + 1]) ++b;
        rows.emplace_back(b, int(t - d.cu_q[b]), h);
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    std::vector<RefRow> out(rows.size());
    const int gqa = p.hq / p.hkv, D = p.d;
    parallel_for(int64_t(rows.size()), [&](int64_t r) {
        auto [b, i, h] = rows[r];
        RefRow& rr = out[r];
        rr.b = b, rr.i = i, rr.h = h;
        rr.o.assign(D, 0.0f);
        const int kh = h / gqa, n = p.kv_end(b, i);
        const uint16_t* qp = &d.q[((int64_t(d.cu_q[b]) + i) * p.hq + h) * D];
        std::vector<float> qf(D);
        for (int x = 0; x < D; ++x) qf[x] = bf2f(qp[x]);
        std::vector<float> s(n);
        float m = -INFINITY;
        for (int j = 0; j < n; ++j) {
            const uint16_t* kp = &d.k[((int64_t(d.cu_k[b]) + j) * p.hkv + kh) * D];
            float acc = 0;
            for (int x = 0; x < D; ++x) acc += qf[x] * bf2f(kp[x]);
            s[j] = acc * p.scale;
            m = std::max(m, s[j]);
        }
        if (p.sink) m = std::max(m, d.sink[h]);
        double l = 0;
        for (int j = 0; j < n; ++j) {
            const float e = std::exp(s[j] - m);
            l += e;
            const uint16_t* vp = &d.v[((int64_t(d.cu_k[b]) + j) * p.hkv + kh) * D];
            for (int x = 0; x < D; ++x) rr.o[x] += e * bf2f(vp[x]);
        }
        if (p.sink) l += std::exp(d.sink[h] - m);
        for (int x = 0; x < D; ++x) rr.o[x] = l > 0 ? float(rr.o[x] / l) : 0.0f;
        rr.lse = l > 0 ? float(m + std::log(l)) : -INFINITY;
    });
    return out;
}

struct DevBufs {
    void *q, *k, *v, *o, *lse, *cuq, *cuk, *sink;
};

// A kernel to run: aiter's asm code object (fn) or the opus kernel of the traits.
struct Kernel {
    std::string name;
    hipFunction_t fn = nullptr;
    int threads = 0, q_tile = 0;  // asm only
    bool lse = true;              // writes the LSE
};

// aiter's asm code object for a problem, run on the varlen kernel arguments; dir is relative to aiter_dir().
struct AsmKernel {
    const char *dir, *co, *sym;
    int threads, q_tile;
};
bool find_asm(const Problem& p, AsmKernel& a) {
#if defined(GQA_ARCH_GFX1250)
    if (p.d != 128) return false;
    a = p.causal ? AsmKernel{"hsa/gfx1250/fmha_fwd_bf16_varlen", "fmha_bf16_pertokenBf16_hd128_128x256_mask_varlen.co",
                             "_ZN5aiter48fmha_bf16_pertokenBf16_hd128_128x256_mask_varlenE", 128, 128}
                 : AsmKernel{"hsa/gfx1250/fmha_fwd_bf16_varlen", "fmha_bf16_pertokenBf16_hd128_128x256_varlen.co",
                             "_ZN5aiter43fmha_bf16_pertokenBf16_hd128_128x256_varlenE", 128, 128};
    return true;
#else
    (void)p;
    (void)a;
    return false;
#endif
}

dim3 varlen_grid(const Problem& p, int q_tile, int opt) {
    const int tiles = (p.max_q() + q_tile - 1) / q_tile;
    int gx = (opt & OPUS_GQA_OPT_DOUBLE_Q) ? (tiles + 1) / 2 : tiles;
    int gy = p.hq, gz = p.batch();
    if (opt & OPUS_GQA_OPT_REMAP_XY) std::swap(gx, gy);
    return dim3(gx, gy, gz);
}

opus_gqa_varlen_kargs varlen_kargs(const Problem& p, const DevBufs& b, int opt) {
    opus_gqa_varlen_kargs a{};
    a.d_addr = b.o;
    a.q_addr = b.q;
    a.k_addr = b.k;
    a.v_addr = b.v;
    a.lse_addr = b.lse;
    a.qseq_addr = b.cuq;
    a.kseq_addr = b.cuk;
    a.scalar = p.scale;
    a.gqa = p.hq / p.hkv;
    a.q_head_num = p.hq;
    a.opt = opt;
    a.lse = 1;
    a.max_q_len = p.max_q();
    a.sink_addr = p.sink ? b.sink : nullptr;
    return a;
}

// [B, N, H, D] with strides: the buffers of a problem with one length N for every batch.
[[maybe_unused]] opus_gqa_kargs fixed_kargs(const Problem& p, const DevBufs& b) {
    const int n = p.sq[0];
    opus_gqa_kargs a{};
    a.ptr_q = b.q;
    a.ptr_k = b.k;
    a.ptr_v = b.v;
    a.ptr_o = b.o;
    a.B = p.batch();
    a.N = n;
    a.H = p.hq;
    a.H_KV = p.hkv;
    a.D = p.d;
    a.stride_q_b = n * p.hq * p.d;
    a.stride_q_n = p.hq * p.d;
    a.stride_q_h = p.d;
    a.stride_kv_b = n * p.hkv * p.d;
    a.stride_kv_n = p.hkv * p.d;
    a.stride_kv_h = p.d;
    return a;
}

// Why the opus kernel cannot run a problem, from the limits its traits declare; empty if it can.
template <class T>
std::string unsupported(const Problem& p) {
    if (!T::VARLEN && !p.fixed_len()) return "needs one length for every batch and sq == sk";
    if (!T::HAS_SINK && p.sink) return "has no sink";
    for (int n : p.sq)
        if (n % T::SEQ_ALIGN || n < T::MIN_KV_TILES * T::KV_TILE_SIZE)
            return "needs lengths that are multiples of " + std::to_string(T::SEQ_ALIGN) + " and at least " +
                   std::to_string(T::MIN_KV_TILES * T::KV_TILE_SIZE);
    return "";
}

template <class T>
void launch(const Kernel& k, const Problem& p, const DevBufs& b, int opt, hipStream_t s) {
    if (k.fn) {
        opus_gqa_varlen_kargs a = varlen_kargs(p, b, opt);
        const dim3 g = varlen_grid(p, k.q_tile, opt);
        size_t sz = sizeof(a);
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        CHECK_HIP(hipModuleLaunchKernel(k.fn, g.x, g.y, g.z, k.threads, 1, 1, 0, s, nullptr, cfg));
        return;
    }
    typename T::kargs_type a;
    dim3 g;
    if constexpr (std::is_same_v<typename T::kargs_type, opus_gqa_varlen_kargs>) {
        a = varlen_kargs(p, b, opt);
        g = varlen_grid(p, T::WG_Q_ROWS, opt);
    } else {
        a = fixed_kargs(p, b);
        g = dim3(p.hq, (p.sq[0] + T::WG_Q_ROWS - 1) / T::WG_Q_ROWS, p.batch());
    }
    if constexpr (T::D_TILE_SIZE == 128) gqa_d128_kernel<T><<<g, T::BLOCK_SIZE, 0, s>>>(a);
#if defined(GQA_ARCH_GFX950)
    else gqa_d512_kernel<T><<<g, T::BLOCK_SIZE, 0, s>>>(a);
#endif
    CHECK_HIP(hipGetLastError());
}

struct Check {
    double maxdiff = 0, lse_maxdiff = 0;
    int64_t bad = 0, n = 0, nonfinite = 0;
    bool pass(double tol) const { return nonfinite == 0 && bad <= int64_t(tol * n); }
};

Check check_ref(const Problem& p, const HostData& d, const std::vector<RefRow>& ref, const std::vector<uint16_t>& o,
                const std::vector<float>& lse, bool with_lse) {
    Check c;
    for (const RefRow& r : ref) {
        const int64_t row = (int64_t(d.cu_q[r.b]) + r.i) * p.hq + r.h;
        for (int x = 0; x < p.d; ++x) {
            const float g = bf2f(o[row * p.d + x]), e = r.o[x];
            if (!std::isfinite(g)) ++c.nonfinite;
            const double df = std::fabs(double(g) - e);
            c.maxdiff = std::max(c.maxdiff, df);
            c.bad += df > 1e-2 + 1e-2 * std::fabs(e);
            ++c.n;
        }
        if (!with_lse) continue;
        const float gl = lse[row];
        if (std::isinf(r.lse) ? !(std::isinf(gl) && gl < 0) : std::fabs(gl - r.lse) > 1e-3 + 1e-3 * std::fabs(r.lse)) ++c.bad;
        if (!std::isinf(r.lse)) c.lse_maxdiff = std::max(c.lse_maxdiff, double(std::fabs(gl - r.lse)));
        ++c.n;
    }
    return c;
}

Check check_pair(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const std::vector<float>& la,
                 const std::vector<float>& lb, bool with_lse) {
    Check c;
    for (size_t i = 0; i < a.size(); ++i) {
        const float x = bf2f(a[i]), y = bf2f(b[i]);
        if (!std::isfinite(x)) ++c.nonfinite;
        const double df = std::fabs(double(x) - y);
        c.maxdiff = std::max(c.maxdiff, df);
        c.bad += df > 1e-2 + 1e-2 * std::fabs(y);
        ++c.n;
    }
    for (size_t i = 0; with_lse && i < la.size(); ++i) {
        if (std::isinf(la[i]) && std::isinf(lb[i])) continue;
        const double df = std::fabs(double(la[i]) - lb[i]);
        c.lse_maxdiff = std::max(c.lse_maxdiff, df);
    }
    return c;
}

void usage() {
    printf(
        "gqa_fwd [-b B -n N | -sq l0,l1,..] [-sk l0,l1,..|-nk N] [-h_q H] [-h_kv H] [-d D] [--causal|--no-causal]\n"
        "        [--sink] [-qscale F] [-opt X] [-kernel K[,K..]] [--verify] [--perf] [-iters N] [-warmup N]\n"
        "        [-asm_dir DIR] [-seed S] [-ref_budget F]\n"
        "  kernels: opus, asm (aiter's code object; gfx1250 D=128)\n");
}

template <class T>
int run(const Problem& p, const Options& o) {
    std::vector<Kernel> kernels;
    hipModule_t mod = nullptr;
    for (size_t s = 0, e; s <= o.kernels.size(); s = e + 1) {
        e = o.kernels.find(',', s);
        if (e == std::string::npos) e = o.kernels.size();
        if (e == s) continue;
        Kernel k;
        k.name = o.kernels.substr(s, e - s);
        if (k.name == "asm") {
            AsmKernel a;
            if (!find_asm(p, a)) {
                fprintf(stderr, "no asm code object for this arch and D=%d\n", p.d);
                return 2;
            }
            const std::string dir = o.asm_dir.empty() ? aiter_dir() + "/" + a.dir : o.asm_dir;
            CHECK_HIP(hipModuleLoad(&mod, (dir + "/" + a.co).c_str()));
            CHECK_HIP(hipModuleGetFunction(&k.fn, mod, a.sym));
            k.threads = a.threads;
            k.q_tile = a.q_tile;
        } else if (k.name == "opus") {
            const std::string why = unsupported<T>(p);
            if (!why.empty()) {
                fprintf(stderr, "the opus kernel %s\n", why.c_str());
                return 2;
            }
            k.lse = T::HAS_LSE;
        } else {
            fprintf(stderr, "unknown kernel %s\n", k.name.c_str());
            usage();
            return 2;
        }
        kernels.push_back(k);
    }
    if (kernels.empty()) { usage(); return 2; }

    const HostData hd = make_data(p);
    const int64_t tq = p.total_q(), tk = p.total_k();
    const size_t qbytes = tq * p.hq * p.d * 2, kbytes = tk * p.hkv * p.d * 2;
    // Rotate input sets so timed launches do not hit a warm last-level cache.
    const int sets = o.perf ? int(std::clamp<int64_t>((1ll << 30) / int64_t(2 * qbytes + 2 * kbytes + 1), 1, 8)) : 1;
    std::vector<DevBufs> bufs(sets);
    for (auto& x : bufs) {
        CHECK_HIP(hipMalloc(&x.q, qbytes));
        CHECK_HIP(hipMalloc(&x.k, kbytes));
        CHECK_HIP(hipMalloc(&x.v, kbytes));
        CHECK_HIP(hipMalloc(&x.o, qbytes));
        CHECK_HIP(hipMalloc(&x.lse, tq * p.hq * 4));
        CHECK_HIP(hipMalloc(&x.cuq, hd.cu_q.size() * 4));
        CHECK_HIP(hipMalloc(&x.cuk, hd.cu_k.size() * 4));
        CHECK_HIP(hipMalloc(&x.sink, p.hq * 4));
        CHECK_HIP(hipMemcpy(x.q, hd.q.data(), qbytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(x.k, hd.k.data(), kbytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(x.v, hd.v.data(), kbytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(x.cuq, hd.cu_q.data(), hd.cu_q.size() * 4, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(x.cuk, hd.cu_k.data(), hd.cu_k.size() * 4, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(x.sink, hd.sink.data(), p.hq * 4, hipMemcpyHostToDevice));
    }

    std::string lens;
    for (int i = 0; i < p.batch() && i < 8; ++i) lens += (i ? "," : "") + std::to_string(p.sq[i]);
    if (p.batch() > 8) lens += ",...";
    printf("CASE b=%d sq=[%s] max_q=%d total_k=%lld h_q=%d h_kv=%d d=%d causal=%d sink=%d opt=0x%x qscale=%g\n",
           p.batch(), lens.c_str(), p.max_q(), (long long)tk, p.hq, p.hkv, p.d, p.causal, p.sink, o.opt, p.qscale);

    hipStream_t s;
    CHECK_HIP(hipStreamCreate(&s));
    std::vector<std::vector<uint16_t>> outs(kernels.size());
    std::vector<std::vector<float>> lses(kernels.size());
    std::vector<RefRow> ref;
    if (o.verify) ref = cpu_reference(p, hd, o.ref_budget);
    bool all_pass = true;
    for (size_t ki = 0; ki < kernels.size(); ++ki) {
        const Kernel& k = kernels[ki];
        CHECK_HIP(hipMemset(bufs[0].o, 0xff, qbytes));
        CHECK_HIP(hipMemset(bufs[0].lse, 0xff, tq * p.hq * 4));
        launch<T>(k, p, bufs[0], o.opt, s);
        CHECK_HIP(hipStreamSynchronize(s));
        outs[ki].resize(tq * p.hq * p.d);
        lses[ki].resize(tq * p.hq);
        CHECK_HIP(hipMemcpy(outs[ki].data(), bufs[0].o, qbytes, hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(lses[ki].data(), bufs[0].lse, tq * p.hq * 4, hipMemcpyDeviceToHost));
        if (!o.verify) continue;
        if (k.fn && p.sink) {  // the asm code objects are built without the sink
            printf("VERIFY %s skipped: the asm code object ignores the sink\n", k.name.c_str());
            continue;
        }
        const Check c = check_ref(p, hd, ref, outs[ki], lses[ki], k.lse);
        const bool ok = c.pass(0.01);
        all_pass &= ok;
        printf("VERIFY %s rows=%zu maxdiff=%.4g lse_maxdiff=%.4g mismatch=%.4f%% nonfinite=%lld %s\n", k.name.c_str(),
               ref.size(), c.maxdiff, c.lse_maxdiff, 100.0 * c.bad / std::max<int64_t>(1, c.n), (long long)c.nonfinite,
               ok ? "PASS" : "FAIL");
    }
    for (size_t x = 0; x < kernels.size(); ++x)
        for (size_t y = 0; y < x; ++y)
            if (!(kernels[y].fn && p.sink)) {
                const Check c = check_pair(outs[x], outs[y], lses[x], lses[y], kernels[x].lse && kernels[y].lse);
                printf("AB %s_vs_%s maxdiff=%.4g lse_maxdiff=%.4g mismatch=%.4f%% nonfinite=%lld\n", kernels[x].name.c_str(),
                       kernels[y].name.c_str(), c.maxdiff, c.lse_maxdiff, 100.0 * c.bad / std::max<int64_t>(1, c.n),
                       (long long)c.nonfinite);
            }
    if (o.perf) {
        const double flops = 4.0 * p.pairs() * p.hq * p.d;
        std::vector<double> us(kernels.size());
        hipEvent_t e0, e1;
        CHECK_HIP(hipEventCreate(&e0));
        CHECK_HIP(hipEventCreate(&e1));
        for (size_t ki = 0; ki < kernels.size(); ++ki) {
            for (int i = 0; i < o.warmup; ++i) launch<T>(kernels[ki], p, bufs[i % sets], o.opt, s);
            CHECK_HIP(hipEventRecord(e0, s));
            for (int i = 0; i < o.iters; ++i) launch<T>(kernels[ki], p, bufs[i % sets], o.opt, s);
            CHECK_HIP(hipEventRecord(e1, s));
            CHECK_HIP(hipEventSynchronize(e1));
            float ms = 0;
            CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
            us[ki] = 1e3 * ms / o.iters;
            printf("PERF %s us=%.2f tflops=%.1f\n", kernels[ki].name.c_str(), us[ki], flops / (us[ki] * 1e6));
        }
        for (size_t ki = 1; ki < kernels.size(); ++ki)
            printf("RATIO %s/%s=%.4f\n", kernels[ki].name.c_str(), kernels[0].name.c_str(), us[ki] / us[0]);
    }
    if (mod) CHECK_HIP(hipModuleUnload(mod));
    return o.verify && !all_pass ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    Problem p;
    Options o;
#if defined(GQA_ARCH_GFX1250)
    o.kernels = "asm,opus";
#else
    o.kernels = "opus";
#endif
    int b = 1, n = 1024, nk = -1;
    std::vector<int> sq, sk;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : (usage(), exit(2), ""); };
        if (!strcmp(a, "-b")) b = atoi(next());
        else if (!strcmp(a, "-n")) n = atoi(next());
        else if (!strcmp(a, "-nk")) nk = atoi(next());
        else if (!strcmp(a, "-sq")) sq = parse_list(next());
        else if (!strcmp(a, "-sk")) sk = parse_list(next());
        else if (!strcmp(a, "-h_q")) p.hq = atoi(next());
        else if (!strcmp(a, "-h_kv")) p.hkv = atoi(next());
        else if (!strcmp(a, "-d")) p.d = atoi(next());
        else if (!strcmp(a, "--causal")) p.causal = true;
        else if (!strcmp(a, "--no-causal")) p.causal = false;
        else if (!strcmp(a, "--sink")) p.sink = true;
        else if (!strcmp(a, "-qscale")) p.qscale = float(atof(next()));
        else if (!strcmp(a, "-opt")) o.opt = int(strtol(next(), nullptr, 0));
        else if (!strcmp(a, "-kernel")) o.kernels = next();
        else if (!strcmp(a, "--verify")) o.verify = true;
        else if (!strcmp(a, "--perf")) o.perf = true;
        else if (!strcmp(a, "-iters")) o.iters = atoi(next());
        else if (!strcmp(a, "-warmup")) o.warmup = atoi(next());
        else if (!strcmp(a, "-asm_dir")) o.asm_dir = next();
        else if (!strcmp(a, "-seed")) p.seed = strtoull(next(), nullptr, 0);
        else if (!strcmp(a, "-ref_budget")) o.ref_budget = atof(next());
        else { usage(); return 2; }
    }
    if (sq.empty()) sq.assign(b, n);
    if (sk.empty()) sk = nk > 0 ? std::vector<int>(sq.size(), nk) : sq;
    if (sk.size() != sq.size()) { fprintf(stderr, "-sq and -sk need the same batch\n"); return 2; }
    p.sq = sq;
    p.sk = sk;
    p.scale = 1.0f / std::sqrt(float(p.d));
    if (p.hq % p.hkv) { fprintf(stderr, "h_q must be a multiple of h_kv\n"); return 2; }

#define RUN_CASE(Q, KV, D, NW) \
    (p.causal ? run<opus_gqa_traits<Q, KV, D, NW, true>>(p, o) : run<opus_gqa_traits<Q, KV, D, NW, false>>(p, o))
    if (p.d == 128) return RUN_CASE(32, 64, 128, 8);
#if defined(GQA_ARCH_GFX950)
    if (p.d == 512) return RUN_CASE(16, 32, 512, 8);
#endif
    fprintf(stderr, "-d %d: no kernel for this arch\n", p.d);
    return 2;
}
