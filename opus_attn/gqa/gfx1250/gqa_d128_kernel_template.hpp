// gfx1250 GQA forward, bf16, D = 128, varlen: 8 waves of 32 Q rows, KV blocks of 64 rows. The waves form two groups of
// four, one wave of each per SIMD, a cluster apart: while one wave of a SIMD reads a K or V block from LDS, the other
// runs WMMAs with the softmax in their shadow.
//
// Per wave and KV block (C layout: lane l holds Q row l % 16, so the softmax is per lane):
//   S^T = K Q^T      A = K from LDS, B = Q in VGPRs
//   P = exp2(S c - m), packed to bf16 in place: two C tiles of 16 KV rows are one B operand of 32
//   O^T += V^T P^T   A = V from LDS by ds_load_tr16_b128, B = P
#pragma once

#include <opus/opus.hpp>
#include "gqa_traits.h"

#define GQA_INLINE __device__ __attribute__((always_inline)) inline

namespace gqa_d128 {

using opus::number;

// ---- Types and helpers ----
using f32     = opus::fp32_t;
using u32     = opus::u32_t;
using u64     = opus::u64_t;
using bf16    = opus::bf16_t;
using f32x2   = opus::fp32x2_t;
using f32x8   = opus::fp32x8_t;
using bf16x8  = opus::bf16x8_t;
using bf16x16 = opus::bf16x16_t;

constexpr float LOG2E = 1.44269504088896340736f, LN2 = 0.69314718055994530942f;
// Finite: exp2(x - M_INIT) of a masked x is 0, not NaN, and a row without keys ends with l = 0.
constexpr float M_INIT = -1.0e30f;

GQA_INLINE int sfl(int x) { return __builtin_amdgcn_readfirstlane(x); }
GQA_INLINE u32 lane_id() { return __builtin_amdgcn_mbcnt_lo(~0u, 0u); }
GQA_INLINE u32 simd_id() { return (__builtin_amdgcn_s_getreg((23) | (0 << 6) | (31 << 11)) >> 8) & 3u; }  // HW_ID1
GQA_INLINE f32 fmax_num(f32 a, f32 b) { return __builtin_fmaxf(a, b); }
GQA_INLINE f32 xhalf(f32 x) {  // lane l ^ 16
    return __builtin_bit_cast(f32, __builtin_amdgcn_permlanex16(__builtin_bit_cast(u32, x), __builtin_bit_cast(u32, x),
                                                                0x76543210u, 0xfedcba98u, false, false));
}
// s_load through the constant address space: data the kernel never writes.
template <class T>
OPUS_D T sload(u64 addr, int idx) {
    return reinterpret_cast<const __attribute__((address_space(4))) T*>(addr)[idx];
}
GQA_INLINE void mem_fence() { asm volatile("" ::: "memory"); }
GQA_INLINE void sched_fence() { __builtin_amdgcn_sched_barrier(0); }

GQA_INLINE bf16x16 cat(bf16x8 a, bf16x8 b) {
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
GQA_INLINE bf16x8 lds_load(u32 addr) { return opus::smem<bf16>{nullptr}.template _load<8>(int(addr)); }
GQA_INLINE bf16x8 lds_load_tr(u32 addr) { return opus::smem<bf16>{nullptr}.template _tr_load<8>(int(addr)); }

// ---- Geometry ----
using cfg = opus_gqa_traits<>;  // the gfx1250 traits allow one tiling
constexpr int D = cfg::D_TILE_SIZE;
constexpr int WAVES = cfg::NUM_WARPS, Q_TILE = cfg::WG_Q_ROWS;
constexpr int KVS = cfg::KV_TILE_SIZE;
constexpr int NB = KVS / 16, NC = KVS / 32;  // S tiles of 16 KV rows, P chunks of 32
constexpr int RING = 4;                      // LDS slots: block b in slot b % RING
constexpr int TAU = 8;                       // O is rescaled once a row max grows by more than 2^TAU
constexpr int EARLY = 4;                     // exp-pass units of a block done in the PV cluster before its QK
constexpr int UNITS = 2 * NB;                // exp-pass units of a block: (Q block, S tile)
constexpr int K_ROW = D * 2 + 16, V_ROW = D * 2 + 32;  // padded LDS rows, V's for the transposed reads
// A slot holds a block as two halves of 32 rows in regions 128 KB apart (different LDS partitions); a wave reads half
// r from region r ^ (SIMD & 1), so that the two SIMD pairs read different partitions.
constexpr int HALF = KVS / 2, REGION = 128 * 1024;
constexpr int K_HALF = HALF * K_ROW, SLOT = HALF * (K_ROW + V_ROW);
constexpr int RING_BYTES = REGION + RING * SLOT;
// After the ring: each wave's Q rows of the workgroup's second tile, staged during the first.
constexpr int Q_STAGE = (RING_BYTES + 255) / 256 * 256, Q_STAGE_WAVE = 32 * K_ROW;
constexpr int LDS_BYTES = Q_STAGE + WAVES * Q_STAGE_WAVE;
constexpr int TDM_ROWS = KVS / WAVES;  // rows of each block a wave loads
static_assert(RING * SLOT <= REGION && LDS_BYTES <= 320 * 1024, "LDS");

// ---- Register tiles and GEMMs ----
using wmma_bf16 = opus::wmma_f32_16x16x32_bf16_w32;

struct q_tile {  // B: Q block j (16 rows) x d chunk c (32)
    bf16x16 v[2][4];
};
struct s_tile {  // S^T: KV tile b (16 rows) x Q block j
    f32x8 v[NB][2];
};
struct o_tile {  // O^T: d block b (16) x Q block j
    f32x8 v[8][2];
};
struct p_tile {  // B: KV chunk c (32 rows) x Q block j
    bf16x16 v[NC][2];
};
struct row_state {  // per Q block: running max (scaled log2) and this lane's partial row sum
    f32 m[2], l[2];
};
struct block_addr {  // a lane's LDS address of each half of a block
    u32 half[2];
};

struct k_block {  // A of QK: KV tile b (16 rows) x d chunk c (32)
    bf16x16 A[NB][4];
    GQA_INLINE void load(const block_addr& a) {
        opus::static_ford<NB, 4>([&](auto b, auto c) {
            const u32 p = a.half[b.value / 2] + u32(b.value % 2 * 16 * K_ROW) + c.value * 64;
            A[b.value][c.value] = cat(lds_load(p), lds_load(p + 32));
        });
    }
};
struct v_block {  // A of PV: d block b (16) x KV chunk c (32 rows)
    bf16x16 A[8][NC];
    GQA_INLINE void load(const block_addr& a) {
        opus::static_ford<NC, 8>([&](auto c, auto b) {
            const u32 p = a.half[c.value] + b.value * 32;
            A[b.value][c.value] = cat(lds_load_tr(p), lds_load_tr(p + 16 * V_ROW));
        });
    }
};

// A GEMM is four groups of 8 WMMAs; extra(g) emits softmax work into group g, 2 VALU and 1 TRANS slots per WMMA.
constexpr int GROUPS = 4;
GQA_INLINE void group_pattern() {
    opus::static_for<8>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // WMMA
        __builtin_amdgcn_sched_group_barrier(0x002, 2, 0);  // VALU
        __builtin_amdgcn_sched_group_barrier(0x400, 1, 0);  // TRANS
    });
}
// QK group g: KV tiles 2 (g / 2) .. +1 x d chunks 2 (g % 2) .. +1
template <class X>
GQA_INLINE void gemm_qk(s_tile& S, const q_tile& Q, const k_block& K, X&& extra) {
    opus::static_for<GROUPS>([&](auto g) {
        sched_fence();
        opus::static_ford<2, 2, 2>([&](auto cc, auto t, auto j) {
            constexpr int b = g.value / 2 * 2 + t.value, c = g.value % 2 * 2 + cc.value;
            if constexpr (c == 0) S.v[b][j.value] = wmma_bf16{}(K.A[b][c], Q.v[j.value][0], f32x8{});
            else S.v[b][j.value] = wmma_bf16{}(K.A[b][c], Q.v[j.value][c], S.v[b][j.value]);
        });
        extra(g);
        group_pattern();
        sched_fence();
    });
}
// PV group g: KV chunk g / 2 x d blocks 4 (g % 2) .. +3
template <class X>
GQA_INLINE void gemm_pv(o_tile& O, const p_tile& P, const v_block& V, X&& extra) {
    opus::static_for<GROUPS>([&](auto g) {
        constexpr int c = g.value / 2;
        sched_fence();
        opus::static_ford<4, 2>([&](auto bb, auto j) {
            constexpr int b = g.value % 2 * 4 + bb.value;
            O.v[b][j.value] = wmma_bf16{}(V.A[b][c], P.v[c][j.value], O.v[b][j.value]);
        });
        extra(g);
        group_pattern();
        sched_fence();
    });
}
// Units [U0, U1) dealt to groups [G0, NG) in order.
template <int U0, int U1, int G0, int NG, int G>
constexpr int deal_lo() { return G < G0 ? U0 : U0 + (G - G0) * (U1 - U0) / (NG - G0); }
template <int U0, int U1, int G0, int NG, int G>
constexpr int deal_hi() { return G < G0 ? U0 : U0 + (G - G0 + 1) * (U1 - U0) / (NG - G0); }

// ---- Online softmax ----
// In pieces, each in the shadow of WMMAs: a block's row max and rescale decision in the PV cluster before its QK
// cluster, the exp pass split between those two clusters, O *= alpha in the memory cluster before its PV. O and l are
// rescaled only when a row max of the wave grows by more than TAU; the decision is a select, so that the exp pass
// shares a scheduling region with the WMMAs.
GQA_INLINE f32 row_max(const s_tile& S, int j) {
    f32 t[4] = {-__builtin_inff(), -__builtin_inff(), -__builtin_inff(), -__builtin_inff()};
    opus::static_ford<NB, 8>([&](auto b, auto i) {
        constexpr int k = (b.value * 8 + i.value) % 4;
        t[k] = fmax_num(t[k], S.v[b.value][j][i.value]);
    });
    const f32 x = fmax_num(fmax_num(t[0], t[1]), fmax_num(t[2], t[3]));
    return fmax_num(x, xhalf(x));
}

struct row_update {
    f32 m[2];      // the max the exp pass subtracts
    f32 alpha[2];  // 1 without rescale
    bool rescale;  // wave-uniform
};
GQA_INLINE void update_rows(const s_tile& S, const row_state& rs, f32 c, row_update& ru) {
    const f32 mx[2] = {row_max(S, 0), row_max(S, 1)};
    f32 mc[2];
    bool need = false;
    opus::static_for<2>([&](auto j) {
        mc[j.value] = fmax_num(rs.m[j.value], mx[j.value] * c);
        need |= mc[j.value] - rs.m[j.value] > f32(TAU);
    });
    const bool any = __builtin_amdgcn_ballot_w32(need) != 0;
    opus::static_for<2>([&](auto j) {
        ru.m[j.value] = any ? mc[j.value] : rs.m[j.value];
        ru.alpha[j.value] = any ? __builtin_amdgcn_exp2f(rs.m[j.value] - mc[j.value]) : 1.0f;
    });
    ru.rescale = any;
}

struct exp_pass {
    f32x2 cc, nm[2], sum[2];
    GQA_INLINE void begin(f32 c, const row_update& ru) {
        asm volatile("" : "+s"(c));  // a fresh SGPR value: {c, c} becomes the SGPR operand of v_pk_fma_f32
        cc = f32x2{c, c};
        opus::static_for<2>([&](auto j) {
            nm[j.value] = f32x2{-ru.m[j.value], -ru.m[j.value]};
            sum[j.value] = f32x2{0.0f, 0.0f};
        });
    }
    template <int J, int B>
    GQA_INLINE void tile(s_tile& S) {
        f32x8& s = S.v[B][J];
        opus::static_for<4>([&](auto i) {
            const f32x2 x = __builtin_elementwise_fma(f32x2{s[2 * i.value], s[2 * i.value + 1]}, cc, nm[J]);
            s[2 * i.value] = __builtin_amdgcn_exp2f(x[0]);
            s[2 * i.value + 1] = __builtin_amdgcn_exp2f(x[1]);
        });
    }
    // row sums and bf16 pack of chunk CC
    template <int J, int CC>
    GQA_INLINE void pack(const s_tile& S, p_tile& P) {
        opus::static_for<2>([&](auto t) {
            const f32x8& s = S.v[2 * CC + t.value][J];
            opus::static_for<4>([&](auto i) { sum[J] += f32x2{s[2 * i.value], s[2 * i.value + 1]}; });
        });
        P.v[CC][J] = cat(opus::cast<bf16>(S.v[2 * CC][J]), opus::cast<bf16>(S.v[2 * CC + 1][J]));
    }
    // tile (U / NB, U % NB); a chunk is packed one unit after its second tile, not right after the exps
    template <int U>
    GQA_INLINE void unit(s_tile& S, p_tile& P) {
        tile<U / NB, U % NB>(S);
        if constexpr (U >= 1 && (U - 1) % 2 == 1) pack<(U - 1) / NB, (U - 1) % NB / 2>(S, P);
    }
    GQA_INLINE void flush(const s_tile& S, p_tile& P) { pack<1, NB / 2 - 1>(S, P); }
    GQA_INLINE void end(row_state& rs, const row_update& ru) {
        opus::static_for<2>([&](auto j) {
            rs.l[j.value] = rs.l[j.value] * ru.alpha[j.value] + (sum[j.value][0] + sum[j.value][1]);
            rs.m[j.value] = ru.m[j.value];
        });
    }
};

// ---- Masks ----
// -inf at KV rows >= lim[j] (relative to the block) for the lane's row of Q block j. Tile b holds half (b / 2) ^ sel.
GQA_INLINE void apply_mask(s_tile& S, const int (&lim)[2], int sel) {
    const int sub = int(lane_id() >> 4) * 8;
    opus::static_for<2>([&](auto j) {
        opus::static_for<NB>([&](auto b) {
            const int t = lim[j.value] - sub - (((b.value / 2) ^ sel) * 2 + b.value % 2) * 16;
            opus::static_for<8>([&](auto i) {
                S.v[b.value][j.value][i.value] = i.value < t ? S.v[b.value][j.value][i.value] : -__builtin_inff();
            });
        });
    });
}

struct wave_rows {
    int row0;       // first Q row
    int kv_end;     // blocks from here on are masked for every row of the wave: skipped
    int mask_from;  // blocks reaching past this need the mask
    int lim[2];     // per lane and Q block: end of the row's keys
};
template <bool CAUSAL>
GQA_INLINE wave_rows make_wave_rows(int row0, int sq, int sk) {
    wave_rows w;
    w.row0 = row0;
    const int diag = sk - sq;  // row i sees keys [0, i + diag]
    const int last = __builtin_elementwise_min(row0 + 31, sq - 1);
    if (CAUSAL) {
        w.kv_end = row0 >= sq ? 0 : __builtin_elementwise_max(__builtin_elementwise_min(last + diag + 1, sk), 0);
        w.mask_from = __builtin_elementwise_min(row0 + diag + 1, sk);
    } else {
        w.kv_end = row0 >= sq ? 0 : sk;
        w.mask_from = sk;
    }
    const int r = row0 + int(lane_id() & 15u);
    opus::static_for<2>([&](auto j) {
        w.lim[j.value] = CAUSAL ? __builtin_elementwise_min(r + 16 * j.value + diag + 1, sk) : sk;
    });
    return w;
}

// ---- Global and LDS traffic ----
// Global accesses go through buffers of the sequence's rows: reads past the end return 0, writes there are dropped.
struct seq_ctx {
    int sq, sk, hq;
    u32 q_row_bytes, kv_row_bytes;
    u64 q_seq, o_seq, lse_seq, k_seq, v_seq;
    f32 c, sink_x;
    bool sink, lse;
};

GQA_INLINE void load_q(q_tile& Q, const seq_ctx& sc, int row0) {
    const u32 l = lane_id();
    auto g = opus::make_gmem(reinterpret_cast<const bf16*>(sc.q_seq), u32(sc.sq) * sc.q_row_bytes);
    opus::static_for<2>([&](auto j) {
        const int os = (row0 + 16 * j.value + int(l & 15u)) * int(sc.q_row_bytes) + int(l >> 4) * 16;
        opus::static_for<4>([&](auto c) {
            const bf16x8 lo = g.template _load<8>(os + c.value * 64);
            const bf16x8 hi = g.template _load<8>(os + c.value * 64 + 32);
            Q.v[j.value][c.value] = cat(lo, hi);
        });
    });
}
// A wave's 32 Q rows by the TDM into LDS (rows past the end land as zeros), read back in the lane layout of K.
GQA_INLINE void stage_q(u32 lds_rows, const seq_ctx& sc, int row0) {
    using window = opus::tdm<bf16, opus::seq<D, 32>, opus::tdm_traits::padding<bf16, D, 8>>;
    window w;
    const int left = sc.sq - row0;
    w.make(lds_rows, reinterpret_cast<const void*>(sc.q_seq + u64(u32(row0 < sc.sq ? row0 : 0)) * sc.q_row_bytes), D,
           u32(left < 0 ? 0 : left), sc.q_row_bytes / 2);
    w.async_load();
}
GQA_INLINE void load_q_lds(q_tile& Q, u32 lds_rows) {
    const u32 a0 = lds_rows + (lane_id() & 15u) * K_ROW + (lane_id() >> 4) * 16u;
    opus::static_ford<2, 4>([&](auto j, auto c) {
        const u32 a = a0 + u32(j.value * 16 * K_ROW + c.value * 64);
        Q.v[j.value][c.value] = cat(lds_load(a), lds_load(a + 32));
    });
}

// K or V rows of one (batch, KV head), TDM_ROWS per load. The D# is built once; a load sets the global address, the
// rows left (0 past the end: a load that reads nothing) and the LDS address.
template <int PAD>
struct kv_stream {
    using window = opus::tdm<bf16, opus::seq<D, TDM_ROWS>, opus::tdm_traits::padding<bf16, D, PAD>>;
    typename window::descriptor d;
    u64 base;
    u32 row_bytes;
    int nrows;
    GQA_INLINE void init(u64 gbase, int n, u32 stride_elems) {
        window w;
        w.make(0u, reinterpret_cast<const void*>(gbase), D, u32(n), stride_elems);
        d = w.make_descriptor();
        base = gbase;
        row_bytes = stride_elems * 2u;
        nrows = n;
    }
    GQA_INLINE void load(int row, u32 lds) {
        const int left = nrows - row;
        d.set_global_addr(base + u64(u32(row)) * row_bytes);
        d.set_tensor_dim1(u32(left < 0 ? 0 : left));
        d.set_lds_addr(lds);
        window::async_load(d);
    }
};
// Wave g loads rows 8 g .. 8 g + 7 of every block.
struct kv_loader {
    kv_stream<8> ks;
    kv_stream<16> vs;
    int g;
    GQA_INLINE void init(const seq_ctx& sc, int wave) {
        ks.init(sc.k_seq, sc.sk, sc.kv_row_bytes / 2);
        vs.init(sc.v_seq, sc.sk, sc.kv_row_bytes / 2);
        g = wave;
    }
    GQA_INLINE void load(int k0, u32 slot) {
        const int row = k0 + g * TDM_ROWS, o = g * TDM_ROWS % HALF;
        const u32 s = slot + u32(g * TDM_ROWS / HALF * REGION);
        ks.load(row, s + u32(o * K_ROW));
        vs.load(row, s + u32(K_HALF + o * V_ROW));
    }
};

// O / l with the sink, and the LSE (natural log), from the C layout; lanes l and l + 16 write the same LSE.
GQA_INLINE void store_o(const o_tile& O, const row_state& rs, const seq_ctx& sc, int row0) {
    const u32 l = lane_id(), lse_row_bytes = u32(sc.hq) * 4u;
    auto go = opus::make_gmem(reinterpret_cast<const bf16*>(sc.o_seq), u32(sc.sq) * sc.q_row_bytes);
    auto gl = opus::make_gmem(reinterpret_cast<const f32*>(sc.lse_seq), u32(sc.sq) * lse_row_bytes);
    opus::static_for<2>([&](auto j) {
        f32 m = rs.m[j.value], s = rs.l[j.value] + xhalf(rs.l[j.value]), scale = 1.0f;
        if (sc.sink) {  // sink_x: the sink logit, scaled log2
            const f32 mn = fmax_num(m, sc.sink_x);
            scale = __builtin_amdgcn_exp2f(m - mn);
            s = s * scale + __builtin_amdgcn_exp2f(sc.sink_x - mn);
            m = mn;
        }
        const f32 inv = s > 0.0f ? scale * __builtin_amdgcn_rcpf(s) : 0.0f;
        const int row = row0 + 16 * j.value + int(l & 15u);
        const int os = row * int(sc.q_row_bytes) + int(l >> 4) * 16;
        opus::static_for<8>([&](auto b) { go.template _store<8>(opus::cast<bf16>(O.v[b.value][j.value] * inv), os + b.value * 32); });
        if (sc.lse) {
            const f32 lse = s > 0.0f ? (m + __builtin_amdgcn_logf(s)) * LN2 : -__builtin_inff();
            gl.template _store<1>(lse, row * int(lse_row_bytes));
        }
    });
}

// ---- Barriers: one named barrier for the 8 waves ----
}  // namespace gqa_d128

__shared__ __amdgpu_named_workgroup_barrier_t gqa_named_bar[1];

namespace gqa_d128 {

GQA_INLINE void bar_init(int members) { __builtin_amdgcn_s_barrier_init(&gqa_named_bar[0], members); }
// A signal sent before the join takes effect could complete the barrier without this wave: wait until STATE_PRIV bits
// 8:4 hold the barrier's ID (index + 1).
GQA_INLINE void bar_join() {
    __builtin_amdgcn_s_barrier_join(&gqa_named_bar[0]);
    for (;;) {
        u32 id;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_STATE_PRIV, 4, 5)" : "=s"(id));
        if (id == 1u) break;
    }
}
GQA_INLINE void bar_signal() {
    mem_fence();
    __builtin_amdgcn_s_barrier_signal_var(&gqa_named_bar[0], 0);
}
GQA_INLINE void bar_wait() {
    __builtin_amdgcn_s_barrier_wait(1);
    mem_fence();
}
GQA_INLINE void wg_barrier() {
    mem_fence();
    __builtin_amdgcn_s_barrier();
    mem_fence();
}
// f() in a loop that runs once. IR Sink moves side-effect-free instructions to their uses across barriers and
// scheduling fences, but not out of a loop: the softmax work of a GEMM cluster would otherwise land clusters later.
template <class F>
GQA_INLINE void in_own_loop(F&& f) {
    u32 again;
    do {
        f();
        asm volatile("s_mov_b32 %0, 0" : "=s"(again));
    } while (again);
}

// ---- The KV walk of a wave over a tile ----
// Step t is four clusters, each ended by a barrier of the 8 waves:
//   M_K(t + 1)   K of block t + 1 into VGPRs, then the TDM load of block t + RING - 1
//   C_QK(t + 1)  QK(t + 1) into the other S buffer, with the rest of block t's exp pass
//   M_V(t)       V of block t into VGPRs, O *= alpha(t), the mask of block t + 1
//   C_PV(t)      PV(t), with block t + 1's row update and first EARLY exp-pass units
// Waves 4-7 take one barrier more at the start and waves 0-3 one more at the end, so that waves 4-7 run a cluster
// behind and one wave per SIMD reads LDS while the other runs WMMAs. A GEMM cluster waits at its end for the barrier of
// the memory cluster before it; a memory cluster starts after every wave's earlier signals. Block b goes into slot
// b % RING in M_K(b - RING + 2), after the lagging waves' last read of the slot in M_V(b - RING). The TDM loads every
// step (past the tile, a load that reads nothing), so that the wait counts are constants; every wave takes nb + 1 steps.

// The two tiles of a workgroup share one TDM stream: past its own blocks the first loads the second's first RING - 2.
struct tile_link {
    int boff = 0;      // the tile's block j is block boff + j of the stream
    bool pre = false;  // its first RING - 2 blocks were loaded by the tile before
    int nb_next = 0;   // blocks of the tile after
};

template <bool CAUSAL>
GQA_INLINE void kv_walk(const seq_ctx& sc, u32 lds, const q_tile& Q, o_tile& O, row_state& rs, const wave_rows& wr,
                       int g, int nb, const tile_link& L) {
    const bool lag = g >= 4;
    const int nsw = __builtin_elementwise_min((wr.kv_end + KVS - 1) / KVS, nb);  // blocks with keys for the wave
    const u32 kl = (lane_id() & 15u) * K_ROW + (lane_id() >> 4) * 16u;
    const u32 vl = ((lane_id() & 7u) + (lane_id() >> 4) * 8u) * V_ROW + ((lane_id() >> 3) & 1u) * 16u;
    u32 k_base[2], v_base[2];
    {
        const u32 sel = simd_id() & 1u;
        opus::static_for<2>([&](auto r) {
            k_base[r.value] = (u32(r.value) ^ sel) * u32(REGION) + kl;
            v_base[r.value] = (u32(r.value) ^ sel) * u32(REGION) + u32(K_HALF) + vl;
        });
    }
    auto slot_of = [&](int i) { return lds + u32((L.boff + i) % RING) * SLOT; };
    auto k_addr = [&](u32 slot) { return block_addr{{k_base[0] + slot, k_base[1] + slot}}; };
    auto v_addr = [&](u32 slot) { return block_addr{{v_base[0] + slot, v_base[1] + slot}}; };
    auto sync = [&] {
        bar_signal();
        bar_wait();
    };
    auto end_mem = [&] { bar_signal(); };
    auto end_gemm = [&] {
        bar_wait();
        sync();
    };
    auto mask = [&](s_tile& S, int k0) {
        if (__builtin_expect(k0 + KVS > wr.mask_from, 0)) {
            const int lim[2] = {wr.lim[0] - k0, wr.lim[1] - k0};
            apply_mask(S, lim, int(simd_id() & 1u));
        }
    };

    kv_loader ld;
    ld.init(sc, g);
    bar_join();
    // block b of the stream: the tile's, the next tile's first RING - 2, else past the end; selects keep the memory
    // clusters straight-line code
    const int n_next = __builtin_elementwise_min(RING - 2, L.nb_next);
    auto load = [&](int b) {
        int k0 = b < nb ? b * KVS : sc.sk;
        const int j = b - nb;
        k0 = j >= 0 && j < n_next ? j * KVS : k0;
        ld.load(k0, slot_of(b));
    };
    if (L.pre) {
        opus::s_wait_tensorcnt<2 * (RING - 2)>();  // block 0 is the first of the RING - 1 loads the tile before left
    } else {
        opus::static_for<RING - 2>([&](auto b) { load(b.value); });
        opus::s_wait_tensorcnt<2 * (RING - 3)>();
    }
    sync();  // block 0 is in LDS
    if (lag) sync();

    s_tile S[2];
    p_tile P[2];  // P(t) in P[t % 2]: the PV cluster reads one while it packs the next
    row_update ru;
    exp_pass E;
    k_block K;
    v_block V;
    // Compile-time flags, so that no register tile is live through a step that skips it. CUR: S buffer of block t;
    // QK: QK(t + 1); SM: block t's softmax and PV; TAIL: M_V waits for block t + 2 only if it is the tile's.
    auto step = [&](auto CUR, auto QK, auto SM, int t, auto TAIL) {
        constexpr int cur = CUR.value, nxt = 1 - cur;
        constexpr bool qk = QK.value, sm = SM.value;
        // M_K(t + 1): the TDM work while the K reads are in flight
        if constexpr (qk) K.load(k_addr(slot_of(t + 1)));
        sched_fence();
        load(t - 1 + RING);
        opus::s_wait_dscnt<0>();
        end_mem();
        // C_QK(t + 1)
        auto rest = [&](auto gg, auto ng) {
            constexpr int u0 = deal_lo<EARLY, UNITS, 0, ng.value, gg.value>(), u1 = deal_hi<EARLY, UNITS, 0, ng.value, gg.value>();
            opus::static_for<u1 - u0>([&](auto k) { E.template unit<u0 + k.value>(S[cur], P[cur]); });
        };
        in_own_loop([&] {
            if constexpr (qk) {
                gemm_qk(S[nxt], Q, K, [&](auto gg) {
                    if constexpr (sm) rest(gg, number<GROUPS>{});
                });
            } else if constexpr (sm) {
                rest(number<0>{}, number<1>{});
            }
            if constexpr (sm) {
                E.flush(S[cur], P[cur]);
                E.end(rs, ru);
            }
        });
        end_gemm();
        // M_V(t)
        if constexpr (sm) V.load(v_addr(slot_of(t)));
        if (!TAIL.value || t + 2 < nb) opus::s_wait_tensorcnt<2 * (RING - 3)>();
        if constexpr (sm) {
            if (__builtin_expect(ru.rescale, 0))
                opus::static_ford<8, 2>([&](auto b, auto j) { O.v[b.value][j.value] *= ru.alpha[j.value]; });
        }
        if constexpr (qk) mask(S[nxt], (t + 1) * KVS);
        opus::s_wait_dscnt<0>();
        end_mem();
        // C_PV(t): the row update in the first group
        auto head = [&](auto gg, auto ng) {
            if constexpr (gg.value == 0) {
                update_rows(S[nxt], rs, sc.c, ru);
                E.begin(sc.c, ru);
            }
            constexpr int g0 = ng.value > 1 ? 1 : 0;
            constexpr int u0 = deal_lo<0, EARLY, g0, ng.value, gg.value>(), u1 = deal_hi<0, EARLY, g0, ng.value, gg.value>();
            opus::static_for<u1 - u0>([&](auto k) { E.template unit<u0 + k.value>(S[nxt], P[nxt]); });
        };
        in_own_loop([&] {
            if constexpr (sm) {
                gemm_pv(O, P[cur], V, [&](auto gg) {
                    if constexpr (qk) head(gg, number<GROUPS>{});
                });
            } else if constexpr (qk) {
                head(number<0>{}, number<1>{});
            }
        });
        end_gemm();
    };
    constexpr number<0> s0{}, no{};
    constexpr number<1> s1{}, yes{};
    // t = -1: QK(0); full steps while block t + 1 has keys; the last such step without QK; then the steps past the
    // wave's keys
    int t = 0;
    if (nsw > 0) {
        step(s1, yes, no, -1, no);
        for (; t + 2 < nsw; t += 2) {
            step(s0, yes, yes, t, no);
            step(s1, yes, yes, t + 1, no);
        }
        if (t + 1 < nsw) step(s0, yes, yes, t++, yes);
        if (t & 1) step(s1, no, yes, t++, yes);
        else step(s0, no, yes, t++, yes);
    } else {
        step(s1, no, no, -1, yes);
    }
    for (; t < nb; ++t) step(s0, no, no, t, yes);
    if (!lag) sync();
    if (L.nb_next == 0) opus::s_wait_tensorcnt<0>();  // the loads past the end, before the wave ends
}

// ---- The kernel ----
// A workgroup takes tile gx of a (batch, Q head) and, with double Q, tile tiles - 1 - gx after it; the first tile
// stages the second's Q in LDS and loads its first blocks.
template <bool CAUSAL>
GQA_INLINE int tile_blocks(const seq_ctx& sc, int q0) {
    const int kv_end = CAUSAL ? __builtin_elementwise_max(__builtin_elementwise_min(q0 + Q_TILE + sc.sk - sc.sq, sc.sk), 0) : sc.sk;
    return (kv_end + KVS - 1) / KVS;
}

// q0_next >= 0: another tile follows; nb_prev >= 0: the tile after one that had nb_prev blocks
template <bool CAUSAL>
GQA_INLINE void wave_tile(const seq_ctx& sc, u32 lds, int g, int q0, int q0_next, int nb_prev) {
    const wave_rows wr = make_wave_rows<CAUSAL>(q0 + 32 * g, sc.sq, sc.sk);
    const u32 q_stage = lds + u32(Q_STAGE + g * Q_STAGE_WAVE);
    tile_link link;
    q_tile Q;
    if (nb_prev >= 0) {  // the tile before waited for it with its first block
        load_q_lds(Q, q_stage);
        link.boff = nb_prev;
        link.pre = true;
    } else {
        load_q(Q, sc, wr.row0);
    }
    if (q0_next >= 0) {  // before the walk's first loads: the TDM completes its loads in order
        stage_q(q_stage, sc, q0_next + 32 * g);
        link.nb_next = tile_blocks<CAUSAL>(sc, q0_next);
    }
    o_tile O;
    opus::static_ford<8, 2>([&](auto b, auto j) { O.v[b.value][j.value] = f32x8{}; });
    row_state rs = {{M_INIT, M_INIT}, {0.0f, 0.0f}};
    kv_walk<CAUSAL>(sc, lds, Q, O, rs, wr, g, tile_blocks<CAUSAL>(sc, q0), link);
    store_o(O, rs, sc, wr.row0);
}

template <bool CAUSAL>
GQA_INLINE void gqa_fwd(const opus_gqa_varlen_kargs& kargs) {
    __shared__ __attribute__((aligned(1024))) char smem[LDS_BYTES];
    const u32 lds = u32(reinterpret_cast<u64>(smem));
    const int wave = sfl(int(__builtin_amdgcn_workitem_id_x()) >> 5);

    const int opt = kargs.opt;
    int gx = opus::block_id_x(), gy = opus::block_id_y();
    if (opt & OPUS_GQA_OPT_REMAP_XY) {
        const int t = gx;
        gx = gy;
        gy = t;
    }
    const int batch = opus::block_id_z(), head = gy;
    seq_ctx sc;
    sc.hq = kargs.q_head_num;
    const int kv_heads = sc.hq / kargs.gqa, kv_head = head / kargs.gqa;
    const u64 cq = reinterpret_cast<u64>(kargs.qseq_addr), ck = reinterpret_cast<u64>(kargs.kseq_addr);
    const int q_start = sload<int>(cq, batch), k_start = sload<int>(ck, batch);
    sc.sq = sload<int>(cq, batch + 1) - q_start;
    sc.sk = sload<int>(ck, batch + 1) - k_start;
    const int tiles = (sc.sq + Q_TILE - 1) / Q_TILE;
    const bool double_q = opt & OPUS_GQA_OPT_DOUBLE_Q;
    if (gx >= (double_q ? (tiles + 1) >> 1 : tiles)) return;

    sc.q_row_bytes = u32(sc.hq) * (D * 2);
    sc.kv_row_bytes = u32(kv_heads) * (D * 2);
    const u64 q_off = (u64(u32(q_start)) * u32(sc.hq) + u32(head)) * (D * 2);
    sc.q_seq = reinterpret_cast<u64>(kargs.q_addr) + q_off;
    sc.o_seq = reinterpret_cast<u64>(kargs.d_addr) + q_off;
    sc.lse_seq = reinterpret_cast<u64>(kargs.lse_addr) + (u64(u32(q_start)) * u32(sc.hq) + u32(head)) * 4;
    const u64 kv_off = (u64(u32(k_start)) * u32(kv_heads) + u32(kv_head)) * (D * 2);
    sc.k_seq = reinterpret_cast<u64>(kargs.k_addr) + kv_off;
    sc.v_seq = reinterpret_cast<u64>(kargs.v_addr) + kv_off;
    sc.c = kargs.scalar * LOG2E;
    sc.sink = kargs.sink_addr != nullptr;
    sc.sink_x = sc.sink ? sload<f32>(reinterpret_cast<u64>(kargs.sink_addr), head) * LOG2E : 0.0f;
    sc.lse = kargs.lse != 0;

    if (wave == 0) bar_init(WAVES);
    wg_barrier();

    const int rounds = double_q && tiles - 1 - gx != gx ? 2 : 1;
    for (int round = 0; round < rounds; ++round) {
        if (round) wg_barrier();  // every wave is done with the first tile
        const int q0 = (round ? tiles - 1 - gx : gx) * Q_TILE, q0_other = (round ? gx : tiles - 1 - gx) * Q_TILE;
        wave_tile<CAUSAL>(sc, lds, wave, q0, round == 0 && rounds == 2 ? q0_other : -1,
                          round ? tile_blocks<CAUSAL>(sc, q0_other) : -1);
    }
}

}  // namespace gqa_d128

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gqa_d128_kernel(opus_gqa_varlen_kargs kargs) {
    gqa_d128::gqa_fwd<Traits::CAUSAL>(kargs);
}
