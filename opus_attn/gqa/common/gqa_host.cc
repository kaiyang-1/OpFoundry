// Host harness of the GQA forward kernels: varlen inputs, the opus kernel of the arch, a CPU reference and timing.
#include <opus/hip_minimal.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(GQA_ARCH_GFX950)
#include "gfx950/gqa_traits.h"
#elif defined(GQA_ARCH_GFX1250)
#include "gfx1250/gqa_traits.h"
#else
#error "No target arch defined. The Makefile passes GQA_ARCH_<ARCH> from ARCH (e.g. ARCH=gfx950)."
#endif

template <class Traits>
__global__ void gqa_d128_kernel(opus_gqa_varlen_kargs kargs);
template <class Traits>
__global__ void gqa_d512_kernel(opus_gqa_varlen_kargs kargs);  // gfx950

#define CHECK_HIP(x)                                                                                \
    do {                                                                                            \
        hipError_t e_ = (x);                                                                        \
        if (e_ != hipSuccess) {                                                                     \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

namespace {

// ---- Problem and inputs ----

// Q/O [total_q, H, D], K/V [total_k, H_KV, D], bf16; batch b has sq[b] Q rows and sk[b] KV rows.
struct Problem {
    std::vector<int> sq, sk;
    int hq = 64, hkv = 8, d = 128;
    bool causal = true;
    bool sink = false;
    float qscale = 1.0f;  // input Q scale: 8 makes row maxes jump by more than the kernels' rescale threshold
    int opt = OPUS_GQA_OPT_DOUBLE_Q | OPUS_GQA_OPT_REMAP_XY;

    float scale() const { return 1.0f / std::sqrt(float(d)); }
    int batch() const { return int(sq.size()); }
    int64_t total_q() const { int64_t t = 0; for (int x : sq) t += x; return t; }
    int64_t total_k() const { int64_t t = 0; for (int x : sk) t += x; return t; }
    int max_q() const { return *std::max_element(sq.begin(), sq.end()); }
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

struct HostData {
    std::vector<uint16_t> q, k, v;
    std::vector<int> cu_q, cu_k;
    std::vector<float> sink;  // scaled-logit domain
};

HostData make_data(const Problem& p) {
    HostData h;
    h.q.resize(p.total_q() * p.hq * p.d);
    h.k.resize(p.total_k() * p.hkv * p.d);
    h.v.resize(h.k.size());
    parallel_for(int64_t(h.q.size() >> 12) + 1, [&](int64_t c) {
        for (int64_t i = c << 12; i < std::min<int64_t>(h.q.size(), (c + 1) << 12); ++i)
            h.q[i] = f2bf(p.qscale * uniform(1, i));
    });
    parallel_for(int64_t(h.k.size() >> 12) + 1, [&](int64_t c) {
        for (int64_t i = c << 12; i < std::min<int64_t>(h.k.size(), (c + 1) << 12); ++i) {
            h.k[i] = f2bf(uniform(2, i));
            h.v[i] = f2bf(uniform(3, i));
        }
    });
    h.cu_q.assign(1, 0);
    h.cu_k.assign(1, 0);
    for (int b = 0; b < p.batch(); ++b) {
        h.cu_q.push_back(h.cu_q.back() + p.sq[b]);
        h.cu_k.push_back(h.cu_k.back() + p.sk[b]);
    }
    h.sink.resize(p.hq);
    for (int i = 0; i < p.hq; ++i) h.sink[i] = 2.0f * uniform(4, i);
    return h;
}

// ---- CPU reference ----

struct RefRow {
    int64_t row;  // (Q row) * H + head, the row of O [total_q * H, D]
    std::vector<float> o;
    float lse;
};

// Every row for small problems, else an even sample with each batch's first and last rows.
std::vector<RefRow> cpu_reference(const Problem& p, const HostData& d) {
    constexpr double BUDGET = 4e9;  // multiply-adds at D = 128
    const double cost = p.pairs() * p.hq * p.d / 128.0;
    const int64_t total_rows = p.total_q() * p.hq;
    const int64_t want = cost <= BUDGET ? total_rows : std::max<int64_t>(64, int64_t(total_rows * BUDGET / cost));
    const int64_t step = std::max<int64_t>(1, total_rows / want);
    std::vector<int64_t> rows;
    for (int b = 0; b < p.batch(); ++b)
        for (int h = 0; h < p.hq; ++h)
            if (p.sq[b] > 0) {
                rows.push_back(int64_t(d.cu_q[b]) * p.hq + h);
                rows.push_back(int64_t(d.cu_q[b + 1] - 1) * p.hq + h);
            }
    for (int64_t r = 0; r < total_rows; r += step) rows.push_back(r);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

    std::vector<RefRow> out(rows.size());
    const int gqa = p.hq / p.hkv, D = p.d;
    const float scale = p.scale();
    parallel_for(int64_t(rows.size()), [&](int64_t r) {
        RefRow& rr = out[r];
        rr.row = rows[r];
        const int64_t t = rr.row / p.hq;
        const int h = int(rr.row % p.hq), kh = h / gqa;
        const int b = int(std::upper_bound(d.cu_q.begin(), d.cu_q.end(), t) - d.cu_q.begin()) - 1;
        const int n = p.kv_end(b, int(t - d.cu_q[b]));
        rr.o.assign(D, 0.0f);
        std::vector<float> qf(D);
        for (int x = 0; x < D; ++x) qf[x] = bf2f(d.q[rr.row * D + x]);
        std::vector<float> s(n);
        float m = -INFINITY;
        for (int j = 0; j < n; ++j) {
            const uint16_t* kp = &d.k[((int64_t(d.cu_k[b]) + j) * p.hkv + kh) * D];
            float acc = 0;
            for (int x = 0; x < D; ++x) acc += qf[x] * bf2f(kp[x]);
            s[j] = acc * scale;
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

// ---- Device buffers and launch ----

// Bytes of each tensor, and of the GUARD rows after it: NaN after Q, K and V (rows past a sequence read as 0),
// untouched after O and the LSE.
struct Sizes {
    static constexpr int64_t GUARD = 256;
    size_t q, k, lse, q_guard, k_guard, lse_guard;
    explicit Sizes(const Problem& p)
        : q(p.total_q() * p.hq * p.d * 2), k(p.total_k() * p.hkv * p.d * 2), lse(p.total_q() * p.hq * 4),
          q_guard(GUARD * p.hq * p.d * 2), k_guard(GUARD * p.hkv * p.d * 2), lse_guard(GUARD * p.hq * 4) {}
};

struct DevBufs {
    void *q, *k, *v, *o, *lse, *cuq, *cuk, *sink;
};

template <class X>
void* upload(const std::vector<X>& h, size_t guard = 0) {
    void* d;
    const size_t bytes = h.size() * sizeof(X);
    CHECK_HIP(hipMalloc(&d, bytes + guard));
    CHECK_HIP(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice));
    if (guard) CHECK_HIP(hipMemset(static_cast<char*>(d) + bytes, 0xff, guard));
    return d;
}

DevBufs upload(const HostData& h, const Sizes& z) {
    DevBufs b;
    b.q = upload(h.q, z.q_guard);
    b.k = upload(h.k, z.k_guard);
    b.v = upload(h.v, z.k_guard);
    b.cuq = upload(h.cu_q);
    b.cuk = upload(h.cu_k);
    b.sink = upload(h.sink);
    CHECK_HIP(hipMalloc(&b.o, z.q + z.q_guard));
    CHECK_HIP(hipMalloc(&b.lse, z.lse + z.lse_guard));
    return b;
}

template <class T>
void launch(const Problem& p, const DevBufs& b) {
    opus_gqa_varlen_kargs a{};
    a.d_addr = b.o;
    a.q_addr = b.q;
    a.k_addr = b.k;
    a.v_addr = b.v;
    a.lse_addr = b.lse;
    a.qseq_addr = b.cuq;
    a.kseq_addr = b.cuk;
    a.scalar = p.scale();
    a.gqa = p.hq / p.hkv;
    a.q_head_num = p.hq;
    a.opt = p.opt;
    a.lse = 1;
    a.sink_addr = p.sink ? b.sink : nullptr;
    int tiles = (p.max_q() + T::WG_Q_ROWS - 1) / T::WG_Q_ROWS;
    if (p.opt & OPUS_GQA_OPT_DOUBLE_Q) tiles = (tiles + 1) / 2;
    const dim3 g = (p.opt & OPUS_GQA_OPT_REMAP_XY) ? dim3(p.hq, tiles, p.batch()) : dim3(tiles, p.hq, p.batch());
    if constexpr (T::D_TILE_SIZE == 128) gqa_d128_kernel<T><<<g, T::BLOCK_SIZE>>>(a);
    else gqa_d512_kernel<T><<<g, T::BLOCK_SIZE>>>(a);
    CHECK_HIP(hipGetLastError());
}

// ---- Verify and perf ----

// O within 1e-2 + 1e-2 |ref| and the LSE within 1e-3 + 1e-3 |ref| for all but 1% of the values; no write past O or the
// LSE.
template <class T>
bool verify(const Problem& p, const HostData& h, const DevBufs& b, const Sizes& z) {
    const std::vector<RefRow> ref = cpu_reference(p, h);
    CHECK_HIP(hipMemset(b.o, 0xff, z.q + z.q_guard));
    CHECK_HIP(hipMemset(b.lse, 0xff, z.lse + z.lse_guard));
    launch<T>(p, b);
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<uint16_t> o(z.q / 2);
    std::vector<float> lse(z.lse / 4);
    CHECK_HIP(hipMemcpy(o.data(), b.o, z.q, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(lse.data(), b.lse, z.lse, hipMemcpyDeviceToHost));

    double maxdiff = 0, lse_maxdiff = 0;
    int64_t bad = 0, n = 0, nonfinite = 0;
    for (const RefRow& r : ref) {
        for (int x = 0; x < p.d; ++x) {
            const float g = bf2f(o[r.row * p.d + x]), e = r.o[x];
            if (!std::isfinite(g)) ++nonfinite;
            const double df = std::fabs(double(g) - e);
            maxdiff = std::max(maxdiff, df);
            bad += df > 1e-2 + 1e-2 * std::fabs(e);
            ++n;
        }
        const float gl = lse[r.row];
        if (std::isinf(r.lse) ? !(std::isinf(gl) && gl < 0) : std::fabs(gl - r.lse) > 1e-3 + 1e-3 * std::fabs(r.lse)) ++bad;
        if (!std::isinf(r.lse)) lse_maxdiff = std::max(lse_maxdiff, double(std::fabs(gl - r.lse)));
        ++n;
    }
    auto written = [](const void* guard, size_t bytes) {  // bytes changed in guard rows
        std::vector<uint8_t> g(bytes);
        CHECK_HIP(hipMemcpy(g.data(), guard, bytes, hipMemcpyDeviceToHost));
        return int64_t(std::count_if(g.begin(), g.end(), [](uint8_t x) { return x != 0xff; }));
    };
    const int64_t guard = written(static_cast<char*>(b.o) + z.q, z.q_guard) +
                          written(static_cast<char*>(b.lse) + z.lse, z.lse_guard);
    const bool ok = nonfinite == 0 && guard == 0 && bad <= n / 100;
    printf("VERIFY rows=%zu maxdiff=%.4g lse_maxdiff=%.4g mismatch=%.4f%% nonfinite=%lld guard=%lld %s\n", ref.size(),
           maxdiff, lse_maxdiff, 100.0 * bad / std::max<int64_t>(1, n), (long long)nonfinite, (long long)guard,
           ok ? "PASS" : "FAIL");
    return ok;
}

// The launches cycle through the input sets, so that they do not hit a warm last-level cache.
template <class T>
void perf(const Problem& p, const std::vector<DevBufs>& sets) {
    constexpr int WARMUP = 5, ITERS = 30;
    hipEvent_t e0, e1;
    CHECK_HIP(hipEventCreate(&e0));
    CHECK_HIP(hipEventCreate(&e1));
    for (int i = 0; i < WARMUP; ++i) launch<T>(p, sets[i % sets.size()]);
    CHECK_HIP(hipEventRecord(e0));
    for (int i = 0; i < ITERS; ++i) launch<T>(p, sets[i % sets.size()]);
    CHECK_HIP(hipEventRecord(e1));
    CHECK_HIP(hipEventSynchronize(e1));
    float ms = 0;
    CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
    const double us = 1e3 * ms / ITERS;
    printf("PERF us=%.2f tflops=%.1f\n", us, 4.0 * p.pairs() * p.hq * p.d / (us * 1e6));
}

// --verify: the CPU reference check; else timing.
template <class T>
int run(const Problem& p, bool check) {
    const HostData h = make_data(p);
    const Sizes z(p);
    const int nsets = check ? 1 : int(std::clamp<int64_t>((1ll << 30) / int64_t(2 * z.q + 2 * z.k + 1), 1, 8));
    std::vector<DevBufs> sets;
    for (int i = 0; i < nsets; ++i) sets.push_back(upload(h, z));

    printf("CASE b=%d sq=[", p.batch());
    for (int i = 0; i < p.batch() && i < 8; ++i) printf(i ? ",%d" : "%d", p.sq[i]);
    printf("%s] max_q=%d total_k=%lld h_q=%d h_kv=%d d=%d causal=%d sink=%d opt=0x%x qscale=%g\n",
           p.batch() > 8 ? ",..." : "", p.max_q(), (long long)p.total_k(), p.hq, p.hkv, p.d, p.causal, p.sink, p.opt,
           p.qscale);

    if (check) return verify<T>(p, h, sets[0], z) ? 0 : 1;
    perf<T>(p, sets);
    return 0;
}

// ---- Command line ----

void usage() {
    printf(
        "gqa_fwd [-b B -n N | -sq l0,l1,..] [-sk l0,l1,..] [-h_q H] [-h_kv H] [-d D] [--causal|--no-causal] [--sink]\n"
        "        [-qscale F] [-opt X] [--verify]\n");
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

bool parse_args(int argc, char** argv, Problem& p, bool& check) {
    int b = 1, n = 1024;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : (usage(), exit(2), ""); };
        if (!strcmp(a, "-b")) b = atoi(next());
        else if (!strcmp(a, "-n")) n = atoi(next());
        else if (!strcmp(a, "-sq")) p.sq = parse_list(next());
        else if (!strcmp(a, "-sk")) p.sk = parse_list(next());
        else if (!strcmp(a, "-h_q")) p.hq = atoi(next());
        else if (!strcmp(a, "-h_kv")) p.hkv = atoi(next());
        else if (!strcmp(a, "-d")) p.d = atoi(next());
        else if (!strcmp(a, "--causal")) p.causal = true;
        else if (!strcmp(a, "--no-causal")) p.causal = false;
        else if (!strcmp(a, "--sink")) p.sink = true;
        else if (!strcmp(a, "-qscale")) p.qscale = float(atof(next()));
        else if (!strcmp(a, "-opt")) p.opt = int(strtol(next(), nullptr, 0));
        else if (!strcmp(a, "--verify")) check = true;
        else { usage(); return false; }
    }
    if (p.sq.empty()) p.sq.assign(b, n);
    if (p.sk.empty()) p.sk = p.sq;
    if (p.sk.size() != p.sq.size()) { fprintf(stderr, "-sq and -sk need the same batch\n"); return false; }
    if (p.hq % p.hkv) { fprintf(stderr, "h_q must be a multiple of h_kv\n"); return false; }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Problem p;
    bool check = false;
    if (!parse_args(argc, argv, p, check)) return 2;
    if (p.d == 128) return p.causal ? run<gqa_d128_traits<true>>(p, check) : run<gqa_d128_traits<false>>(p, check);
#if defined(GQA_ARCH_GFX950)
    if (p.d == 512) return p.causal ? run<gqa_d512_traits<true>>(p, check) : run<gqa_d512_traits<false>>(p, check);
#endif
    fprintf(stderr, "-d %d: no kernel for this arch\n", p.d);
    return 2;
}
