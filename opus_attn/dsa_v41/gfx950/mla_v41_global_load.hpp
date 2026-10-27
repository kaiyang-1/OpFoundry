#pragma once

#include <opus/opus.hpp>

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#define OPUS_GLOBAL_ADDR __attribute__((address_space(1)))
#endif

template<opus::index_t BYTES, int G_IMM>
__device__ inline void global_load_lds_raw(const char* src, void* lds_ptr) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    __builtin_amdgcn_global_load_lds(
        (OPUS_GLOBAL_ADDR void*)(const_cast<char*>(src)),
        (OPUS_LDS_ADDR void*)(reinterpret_cast<__UINTPTR_TYPE__>(lds_ptr)),
        BYTES, G_IMM, 0);
#else
    __builtin_memcpy(reinterpret_cast<char*>(lds_ptr) + G_IMM, src + G_IMM, BYTES);
#endif
}

template<opus::index_t VEC, int M0_IMM, int G_IMM, class D, class S>
__device__ inline void global_load_lds(const D* src, const S* lds_base) {
    constexpr opus::index_t BYTES = VEC * static_cast<opus::index_t>(sizeof(D));
    static_assert(G_IMM >= 0 && G_IMM < (1 << 12) && G_IMM % 4 == 0);
    static_assert(M0_IMM >= 0 && M0_IMM % 4 == 0);
    global_load_lds_raw<BYTES, G_IMM>(
        reinterpret_cast<const char*>(src),
        reinterpret_cast<char*>(const_cast<S*>(lds_base)) + M0_IMM);
}

// global -> VGPR, one issue per y-position of the layout (drop-in for opus::load(gmem, layout)).
template<opus::index_t VEC, class D, class Layout,
         std::enable_if_t<opus::is_layout_v<Layout>, bool> = true>
__device__ inline auto global_load(const D* g_base, const Layout& u) {
    constexpr opus::index_t N = opus::layout_load_traits<Layout, VEC>::r_elem.value;
    const auto os = opus::layout_to_offsets<VEC>(u);
    opus::vector_t<D, VEC * N> r;
    opus::static_for<N>([&](auto i) {
        opus::set_slice(r, *reinterpret_cast<const opus::vector_t<D, VEC>*>(
                               reinterpret_cast<const char*>(g_base)
                               + static_cast<int64_t>(os[i.value]) * static_cast<int64_t>(sizeof(D))),
                        opus::number<i.value * VEC>{}, opus::number<(i.value + 1) * VEC>{});
    });
    return r;
}

// global -> VGPR at a plain element offset.
template<opus::index_t VEC, class D>
__device__ inline auto global_load(const D* g_base, opus::index_t os) {
    return *reinterpret_cast<const opus::vector_t<D, VEC>*>(
        reinterpret_cast<const char*>(g_base) + static_cast<int64_t>(os) * static_cast<int64_t>(sizeof(D)));
}
