// Paged sparse attention kernel template for D=512 on gfx1250 (wave32 / WMMA).
// 16mx4_64nx1 variant (T_M=4, T_N=1).
//
// Placeholder — no implementation yet.
//
// Include this header from per-variant .cc files that instantiate specific traits.
#pragma once

#include <opus/opus.hpp>
#include "pa_traits.h"

namespace pa_16mx4_64nx1 {

// TODO: implement gfx1250 wave32/WMMA kernel body and layout helpers.

} // namespace pa_16mx4_64nx1
