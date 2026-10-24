# okc: problem=muse_bf16_proj
"""Dense projections: small_m (12-bit lossless or BF16 register-direct GEMV) for the shapes it claims, then the
register-direct swap-AB GEMV for small m, 12-bit lossless pooler weights at m = 1, and a TMA + wgmma router for
64 < m <= 256. Shapes no kernel claims fall back to SGLang's own dispatch."""

import hashlib
import zlib
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config




# ---- small_m/small_m_kernels.py ----


SM_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SMD __device__ __forceinline__

SMD int4 sm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint4 sm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint32_t sm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SMD uint32_t sm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SMD void sm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SMD void sm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = sm_prmt(T0, T1, cm);
  const uint32_t hhi = sm_prmt(T0, T1, cm >> 16);
  a[0] = sm_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = sm_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = sm_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = sm_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void sm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SMD void sm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) sm_never(p);
}

SMD uint32_t sm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SMD uint32_t sm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SMD uint32_t sm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SMD void sm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SMD void sm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SMD void sm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SMD void sm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SMD void sm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SMD void sm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SMD bool sm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SMD uint64_t sm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
SMD void sm_wait_bar(uint32_t bar) {
  if (sm_mbar_try(bar)) return;
  const uint64_t t0 = sm_gtime();
  while (!sm_mbar_try(bar))
    if (sm_gtime() - t0 > 20000000ull) break;
}

SMD void sm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SMD float4 sm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SMD void sm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SMD void sm_store(float* p, float v) { *p = v; }

template <int FMT> struct SmFmt;
template <> struct SmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct SmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct SmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct SmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SMD void sm_decode(const int4 (&w)[SmFmt<FMT>::NL], uint32_t (&A)[SmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < SmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < SmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      sm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      sm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      sm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      sm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SMD void sm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SMD void sm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) sm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
__global__ void __launch_bounds__(NTMAX) sm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = SmFmt<FMT>::NL, NA = SmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = sm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = sm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        sm_mbar_init(bar, 1);
        sm_mbar_expect(bar, parts * RT * TV * 128u);
        sm_fence_init();
      }
      sm_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = sm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = sm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = sm_ldx(xp);
          xv[slot][n][1] = sm_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    sm_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          sm_order_point(m, outp);
        } else {
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) sm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          sm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) sm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  sm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) sm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          sm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) sm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                sm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) sm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) sm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            sm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            sm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) sm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            sm_cl_wait();
            if (t < TV) {
              const uint32_t rbar = sm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else sm_st_async4(sm_mapa(sm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            sm_wait_bar(bar);
            sm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            sm_cl_wait();
            const uint32_t rbar = sm_mapa(bar, 0);
            sm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else sm_st_async4(sm_mapa(sm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            sm_wait_bar(bar);
            sm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int sm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
static int sm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = sm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = sm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
"""

SM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

SM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
SM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
SM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
SM_ESC_CAP = 32


def sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)


def sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7))
        cases.append(f"    case {i}: return sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return SM_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def sm_build(instances, name="small_m_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, SM_CPP + extra_cpp, cuda_sources=sm_cuda_source(instances) + extra_cuda,
                       functions=["sm_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def sm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def _to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def _to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def sm_c12_unpack(data, base, esc):
    """Inverse of sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(SM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def sm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def sm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [sm_units(hi), sm_units(lo)]
    else:
        mats = [sm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = sm_c12_bytes(v)
        if not torch.equal(sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(sm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = sm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > SM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = _to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.sm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


# ---- deliverable API ----
# Exact token counts take precedence over token-tile entries. Geometry and reduction mapping are H100-validated;
# the compressed m=16/32 configurations have not yet been measured on the H200 judge.
SM_TABLE_M = {
    ("router", 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8)),
}
SM_TABLE_MT = {
    ("pooler", 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)),
    ("qkv_gate", 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)),
    ("o", 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("o", 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("unpooler", 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)),
    ("router", 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)),
}
SM_MAX_M = 32

SM_TABLE_M.update({
    ("router", 8): SM_TABLE_MT[("router", 1)],
    ("qkv_gate", 8): SM_TABLE_MT[("qkv_gate", 1)],
    ("pooler", 8): SM_TABLE_MT[("pooler", 1)],
    ("o", 8): SM_TABLE_MT[("o", 1)],
    ("unpooler", 1): SM_TABLE_MT[("unpooler", 1)],
    ("unpooler", 8): SM_TABLE_MT[("unpooler", 1)],
})
SM_TABLE_MT[('pooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT[('pooler', 4)] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT[('qkv_gate', 2)] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
SM_TABLE_MT[('unpooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))


def sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


def sm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = sm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = sm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = sm_verify_onehot(ext, inst_id[inst], sp, sm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


def prepare_small_m(weights, table_m=None, table_mt=None, name="small_m_gemv"):
    table_m = dict(SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if sm_bf16_twin(inst):
            instances.append(sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = sm_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and sm_bf16_twin(inst):
            plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, sm_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    return st


def run_small_m(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= SM_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", sm_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != SM_LAYER_K[layer] or not x.is_cuda:
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    if sm_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out


COMPRESSED_PAIRS = {("pooler", 1)}


# ---- m32/m32_kernels.py ----


M32_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define M32D __device__ __forceinline__

M32D int4 m32_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint4 m32_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint32_t m32_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

M32D uint32_t m32_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

M32D void m32_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
M32D void m32_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = m32_prmt(T0, T1, cm);
  const uint32_t hhi = m32_prmt(T0, T1, cm >> 16);
  a[0] = m32_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = m32_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = m32_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = m32_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void m32_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
M32D void m32_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) m32_never(p);
}

M32D uint32_t m32_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

M32D uint32_t m32_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

M32D uint32_t m32_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

M32D void m32_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

M32D void m32_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

M32D void m32_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
M32D void m32_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
M32D void m32_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

M32D void m32_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

M32D bool m32_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

M32D uint64_t m32_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
M32D void m32_wait_bar(uint32_t bar) {
  if (m32_mbar_try(bar)) return;
  const uint64_t t0 = m32_gtime();
  while (!m32_mbar_try(bar))
    if (m32_gtime() - t0 > 20000000ull) break;
}

M32D void m32_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

M32D float4 m32_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

M32D void m32_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
M32D void m32_store(float* p, float v) { *p = v; }

template <int FMT> struct M32Fmt;
template <> struct M32Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct M32Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct M32Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct M32Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
M32D void m32_decode(const int4 (&w)[M32Fmt<FMT>::NL], uint32_t (&A)[M32Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < M32Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < M32Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      m32_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      m32_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      m32_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      m32_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
M32D void m32_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
M32D void m32_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) m32_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
__global__ void __launch_bounds__(NTMAX) m32_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = M32Fmt<FMT>::NL, NA = M32Fmt<FMT>::NA, MTT = MT * TG;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
  const int wbytes = XM == 4 ? CWR * CWK * RT * NA * 2048 : 0;
  int4* sw = reinterpret_cast<int4*>(smem + xbytes);
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;
    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MTT, (m + 1) >> 1);
    const uint32_t bar = m32_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = m32_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        m32_mbar_init(bar, 1);
        m32_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        m32_fence_init();
      }
      m32_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = m32_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = m32_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = m32_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if (XM != 4 || tg == 0) w[slot][j][l] = m32_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = m32_ldx(xp);
          xv[slot][n][1] = m32_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    m32_order_point(m, outp);
    if constexpr (XM == 3 || XM == 5) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          m32_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
          if constexpr (XM == 7 || XM == 8 || XM == 11) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=m32_ldx(xp), z1=m32_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            m32_order_point(m, outp);
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) m32_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          m32_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) m32_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
          if constexpr (XM == 4) {
            if (tg == 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q)
                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =
                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);
            }
            __syncthreads();
            if (tg != 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q) {
                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];
                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;
                }
            }
          }
          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  m32_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = m32_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  m32_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) m32_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
                }
              }
            }
          } else {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              uint32_t bx[8];
              if constexpr (XM >= 1 && XM <= 4 && !NOX) {
                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);
                const __nv_bfloat16* xp;
                if constexpr (XM == 2 || XM == 3 || XM == 5)
                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;
                else xp = x0 + (size_t)tok * K + si * 64;
                uint4 z0, z1;
                if constexpr (XM == 2 || XM == 3 || XM == 5) {
                  z0 = *reinterpret_cast<const uint4*>(xp);
                  z1 = *reinterpret_cast<const uint4*>(xp + 8);
                } else {
                  z0 = m32_ldx(xp); z1 = m32_ldx(xp + (XM >= 9 ? 32 : 8));
                }
                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;
                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;
              }
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);
                } else {
                  m32_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) m32_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          m32_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) m32_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                m32_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) m32_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) m32_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          const int NI = WR * RT * MTT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            m32_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            m32_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) m32_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2 || RED == 4) {
            m32_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = m32_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = (tg * MT + n) * 4 + t;
                  if (RED == 4 || tn < TV) {
                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else m32_st_async4(m32_mapa(m32_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            m32_wait_bar(bar);
            if constexpr (RED == 4) m32_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else m32_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (RED == 4 || t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if ((tg * MT + n) * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            m32_cl_wait();
            const uint32_t rbar = m32_mapa(bar, 0);
            m32_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else m32_st_async4(m32_mapa(m32_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            m32_wait_bar(bar);
            m32_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int m32_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
static int m32_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = m32_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * TG * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT * TG, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;
    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;
  if (XM == 4) sm += (size_t)WR * WK * RT * M32Fmt<FMT>::NA * 2048;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = m32_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int m32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
"""

M32_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int m32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

M32_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
M32_NL = {0: 4, 1: 3, 2: 8, 3: 6}
M32_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
M32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
M32_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
M32_ESC_CAP = 32


def m32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)


def m32_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,)))
        cases.append(f"    case {i}: return m32_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return M32_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def m32_build(instances, name="m32_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, M32_CPP + extra_cpp, cuda_sources=m32_cuda_source(instances) + extra_cuda,
                       functions=["m32_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def m32_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def m32_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def m32_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def m32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = m32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def m32_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(M32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = m32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def m32_c12_unpack(data, base, esc):
    """Inverse of m32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(M32_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def m32_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def m32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [m32_units(hi), m32_units(lo)]
    else:
        mats = [m32_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([m32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = m32_c12_bytes(v)
        if not torch.equal(m32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(m32_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def m32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def m32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = m32_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, M32_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > M32_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = m32_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def m32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


# H100-selected candidates; H200 confirmation is recorded separately.
M32_TABLE_M = {}
M32_MAX_M = 32
M32_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def m32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


def m32_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def m32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def m32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if m32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def m32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if m32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def m32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def m32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = m32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = m32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = m32_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = m32_verify_onehot(ext, inst_id[inst], sp, m32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            m32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


def prepare_m32(weights, table_m=None, table_mt=None, name="m32_gemv"):
    table_m = dict(M32_TABLE_M if table_m is None else table_m)
    table_mt = dict(M32_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = m32_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, M32_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if m32_bf16_twin(inst):
            instances.append(m32_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = m32_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and m32_bf16_twin(inst):
            plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, m32_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    st["outputs"] = {layer: torch.empty((M32_MAX_M, w.shape[0]), dtype=w.dtype, device=w.device)
                     for layer, w in weights.items() if any(key[0] == layer for key in st["plans"])}
    return st


def run_m32(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= M32_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", m32_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != M32_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=st["outputs"][layer].dtype, device=x.device)
    if m32_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out


# ---- wide2/wide2_kernels.py ----



def mma(n):
    regs=','.join(f'%{i}' for i in range(n//2))
    outs=','.join(f'"+f"(a[{i}])' for i in range(n//2))
    return f'''__device__ void mma{n}(float* a,uint64_t x,uint64_t y) {{
      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(y));
    }}'''


CUDA=r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ uint32_t addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ void waitbar(uint32_t b,int p) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");
}
__device__ void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
__device__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
__device__ void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");
}
MMA_FUNCTIONS
template<int BN,bool PLAIN>
__global__ __launch_bounds__(256,1) void ss_gemm(const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    __nv_bfloat16* out,const uint8_t* w,int m,int n,int k,int split,int stages,int hint) {
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int AB=64*128,SB=(64+BN)*128;
  uint32_t base=addr(sm),full=base+stages*SB,empty=full+stages*8;
  int tid=threadIdx.x,rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;
  int kb0=(k/64)*rank/split,nkb=(k/64)*(rank+1)/split-kb0;
  if(tid==0) {
    for(int s=0;s<stages;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],%2; mbarrier.init.shared::cta.b64 [%1],1;" :: "r"(full+s*8),"r"(empty+s*8),"r"(PLAIN?129:1) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if(tid>=128) {
    if constexpr(PLAIN) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    if(tid==128 || PLAIN) {
      uint64_t pw,px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
      if(!hint) { asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw)); px=pw; }
      for(int i=0;i<nkb;i++) {
        int s=i%stages,ph=(i/stages)&1;
        waitbar(empty+s*8,ph^1);
        if(tid==128) {
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(PLAIN?BN*128:SB) : "memory");
          if constexpr(!PLAIN) tma(base+s*SB,&tw,(kb0+i)*64,nt*64,full+s*8,pw);
          tma(base+s*SB+AB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);
        }
        if constexpr(PLAIN) {
          uint4 v[4];
          const uint8_t* src=w+(long)(nt*(k/64)+kb0+i)*8192;
          #pragma unroll
          for(int j=0;j<4;j++) {
            int q=tid-128+j*128;
            asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v[j].x),"=r"(v[j].y),"=r"(v[j].z),"=r"(v[j].w) : "l"(src+q*16) : "memory");
          }
          #pragma unroll
          for(int j=0;j<4;j++) {
            int q=tid-128+j*128,r=q/8,c=q%8;
            uint32_t dst=base+s*SB+r*128+((c^(r&7))*16);
            asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};" :: "r"(dst),"r"(v[j].x),"r"(v[j].y),"r"(v[j].z),"r"(v[j].w) : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          arrive(full+s*8);
        }
      }
    }
  } else {
    if constexpr(PLAIN) asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
    else asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) a[j]=0;
    for(int i=0;i<nkb;i++) {
      int s=i%stages,ph=(i/stages)&1;
      waitbar(full+s*8,ph);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<4;kk++) {
        if constexpr(BN==64) mma64(a,desc(base+s*SB+kk*32),desc(base+s*SB+AB+kk*32));
        if constexpr(BN==128) mma128(a,desc(base+s*SB+kk*32),desc(base+s*SB+AB+kk*32));
        if constexpr(BN==256) mma256(a,desc(base+s*SB+kk*32),desc(base+s*SB+AB+kk*32));
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      if(i>0 && tid==0) arrive(empty+((i-1)%stages)*8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    int row=(tid/32)*16+(tid%32)/4,col=2*(tid%4);
    // All producer copies are complete when the final MMA finishes.
    asm volatile("bar.sync 1,128;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      #pragma unroll
      for(int h=0;h<2;h++) {
        ((float*)sm)[(col+8*j)*68+row+8*h]=a[4*j+2*h];
        ((float*)sm)[(col+8*j+1)*68+row+8*h]=a[4*j+2*h+1];
      }
    }
  }
  __syncthreads();
  if(split>1) cluster_sync();
  for(int q=tid+rank*256;q<BN*16;q+=256*split) {
    int token=q/16,row=(q%16)*4;
    if(mt*BN+token>=m) continue;
    float4 sum=make_float4(0,0,0,0);
    for(int s=0;s<split;s++) {
      uint32_t remote=base+(token*68+row)*4;
      if(split>1) asm volatile("mapa.shared::cluster.u32 %0,%1,%2;" : "=r"(remote) : "r"(remote),"r"(s));
      float4 v;
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "r"(remote) : "memory");
      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    uint2 pk=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);
    *(uint2*)(out+(long)(mt*BN+token)*n+nt*64+row)=pk;
  }
  __syncthreads();
  if(split>1) cluster_sync();
}
using EncodeFn=decltype(&cuTensorMapEncodeTiled);
int map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {
  static EncodeFn encode=nullptr;
  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}
  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,bool PLAIN> int launch_t(const void* x,const void* w,void* out,int m,int n,int k,int split,int stages,int hint,cudaStream_t st) {
  CUtensorMap tx,tw;int r=map(&tx,x,m,k,BN);if(r)return r;r=map(&tw,w,n,k,64);if(r)return r;
  int smem=max(stages*((64+BN)*128+16),BN*68*4);
  cudaFuncSetAttribute(ss_gemm<BN,PLAIN>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  cudaFuncSetAttribute(ss_gemm<BN,PLAIN>,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/64)*split,(m+BN-1)/BN);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=smem;cfg.stream=st;
  cudaLaunchAttribute a={};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim={unsigned(split),1,1};
  if(split>1) {cfg.attrs=&a;cfg.numAttrs=1;}
  return int(cudaLaunchKernelEx(&cfg,ss_gemm<BN,PLAIN>,tw,tx,(__nv_bfloat16*)out,(const uint8_t*)w,m,n,k,split,stages,hint));
}
int launch_ss(const void* x,const void* w,void* out,int m,int n,int k,int bn,int split,int stages,int hint,int plain,cudaStream_t st) {
  if(bn==64)return launch_t<64,false>(x,w,out,m,n,k,split,stages,hint,st);
  if(bn==128)return launch_t<128,false>(x,w,out,m,n,k,split,stages,hint,st);
  if(bn==256)return launch_t<256,false>(x,w,out,m,n,k,split,stages,hint,st);
  return -1;
}
'''
CPP=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int launch_ss(const void*,const void*,void*,int,int,int,int,int,int,int,int,cudaStream_t);
int launch(torch::Tensor x,torch::Tensor w,torch::Tensor y,int bn,int split,int stages,int hint,int plain) {
  return launch_ss(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.size(0),y.size(1),x.size(1),bn,split,stages,hint,plain,at::cuda::getCurrentCUDAStream());
}
'''


def wide2_build():
    source=CUDA.replace('MMA_FUNCTIONS','\n'.join(mma(n) for n in [64,128,256]))
    return load_inline('wide2_final_v1',CPP,cuda_sources=source,functions=['launch'],
        extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'])


CANDIDATE_CONFIGS = {
    (128, "qkv_gate"): (128, 3, 8, 1),
    (128, "o"): (128, 1, 8, 1),
    (128, "pooler"): (128, 2, 8, 1),
    (128, "unpooler"): (128, 1, 9, 1),
    (256, "qkv_gate"): (256, 3, 5, 1),
    (256, "o"): (256, 1, 5, 1),
    (256, "pooler"): (256, 2, 5, 1),
    (256, "unpooler"): (256, 1, 5, 1),
}


def prepare_wide2(weights) -> dict:
    device = weights["qkv_gate"].device
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        return {"configs": {}}
    h200_keys = [(128, "qkv_gate"), (128, "pooler"), (256, "pooler")]
    keys = h200_keys
    configs = {key: CANDIDATE_CONFIGS[key] for key in keys}
    return {
        "ext": wide2_build(),
        "weights": {name: weight.contiguous() for name, weight in weights.items() if name != "router"},
        "configs": configs,
    }


def run_wide2(x, layer, n, st):
    m = x.shape[0]
    cfg = st["configs"].get((m, layer))
    if cfg is None or m > 256 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    weight = st["weights"].get(layer)
    if weight is None or weight.shape != (n, x.shape[1]) or weight.dtype != torch.bfloat16:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = st["ext"].launch(x, weight, out, *cfg, 0)
    if rc:
        raise RuntimeError(f"wide2 launch failed: {rc}")
    return out


# ---- router/router_kernels.py ----



def _mma(n):
    regs = ', '.join(f'%{i}' for i in range(n // 2))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
}}'''


ROUTER_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t rr_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void rr_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void rr_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t rr_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void rr_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 rr_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void rr_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t rr_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void rr_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}
@@MMA@@

template<int BN,int S,int ST,int SW,int RED,int PIPE>
__global__ __launch_bounds__((PIPE&2)?256:128) void rr_kernel(const __grid_constant__ CUtensorMap tw,
                                               const __grid_constant__ CUtensorMap tx, float* out, int m) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int WB=NR*128, XB=TM*128, SB=WB*2+XB;
  constexpr int NF=64*BN, NV=NF/4, OWN=NV/S;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S;
  const int row=blockIdx.x/S*NR, tok=blockIdx.y*TM;
  const uint32_t base=rr_smem(sm), bars=base+ST*SB, empty=bars+ST*8, rb=(empty+((PIPE&2)?ST*8:0)+15)&~15u;
  const uint32_t recv=rb+16;
  const int kb=rank*(128/S);
  auto load=[&](int i) {
    const int st=i%ST;
    const uint32_t b=bars+st*8, d=base+st*SB;
    rr_expect(b,SB);
    rr_tma(d,&tw,0,((row/NR)*128+kb+i)*NR*2,b);
    rr_tma(d+2*WB,&tx,(kb+i)*64,tok,b);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) { rr_init(bars+i*8,1); if constexpr(PIPE&2) rr_init(empty+i*8,4); }
    asm volatile("fence.proxy.async.shared::cta;" :::"memory");
    if constexpr(PIPE&4) {
    if constexpr(RED>=1) {
      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
#pragma unroll
    for(int i=0;i<ST && i<128/S;i++) load(i);
    if constexpr(!(PIPE&4)) {
    if constexpr(RED>=1) {
      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
  }
  __syncthreads();
  if constexpr(RED>=1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) {
        for(int i=ST;i<128/S;i++) {
          rr_wait(empty+(i%ST)*8,((i/ST)-1)&1);
          load(i);
        }
      }
      if constexpr(RED>=1) asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
      if constexpr(RED!=2) rr_sync();
      return;
    }
  }
  float acc[BN/2];
#pragma unroll
  for(int j=0;j<BN/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(acc[j]));
  int prev=0;
#pragma unroll 1
  for(int i=0;i<128/S;i++) {
    const int st=i%ST;
    rr_wait(bars+st*8,(i/ST)&1);
    const uint32_t w=base+st*SB, x=w+2*WB;
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;" :::"memory");
#pragma unroll
    for(int q=0;q<4;q++) {
      const uint64_t dx=rr_desc(x+q*32),dh=rr_desc(w+q*32),dl=rr_desc(w+WB+q*32);
      if constexpr(SW) { @@CALL_H@@; @@CALL_L@@; }
      else { @@CALL_NH@@; @@CALL_NL@@; }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" :::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" :::"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    if constexpr(PIPE&2) {
      if constexpr(PIPE&1) {
        if(i>0 && prev+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+(prev%ST)*8):"memory");
      } else {
        if(i+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+st*8):"memory");
      }
    } else if constexpr(PIPE&1) {
      if(i>0 && prev+ST<128/S && tid==0) load(prev+ST);
    } else {
      if(i+ST<128/S && tid==0) load(i+ST);
    }
    prev=i;
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
  auto emit=[&](int v,float4 z) {
    const int t=v%128, j=v/128, ln=t&31, wr=t>>5;
    const int ar=wr*16+(ln>>2),bc=j*8+(ln&3)*2;
    const float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      const int a=ar+(c/2)*8, b=bc+(c&1);
      const int tr=tok+(SW?b:a), wc=row+(SW?a:b);
      if(tr<m) out[tr*256+wc]=vv[c];
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int j=0;j<BN/8;j++) emit(j*128+tid,make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]));
  } else if constexpr(RED==0) {
    float4* part=(float4*)sm;
#pragma unroll
    for(int j=0;j<BN/8;j++) part[j*128+tid]=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
    rr_sync();
    for(int t=tid;t<OWN;t+=128) {
      const int v=rank*OWN+t;
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rr_add(sum,rr_pull(rr_map(base+v*16,r)));
      emit(v,sum);
    }
    rr_sync();
  } else {
    asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/8;j++) {
      const int v=j*128+tid, dst=v/OWN, local=v%OWN;
      const float4 val=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
      const uint32_t addr=recv+(rank*OWN+local)*16;
      if(dst==rank) *(float4*)(sm+(addr-base))=val;
      else rr_push(rr_map(addr,dst),val,rr_map(rb,dst));
    }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;" :::"memory"); else __syncthreads();
    rr_wait(rb,0);
    for(int t=tid;t<OWN;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rr_add(sum,*(float4*)(sm+recv-base+(r*OWN+t)*16));
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED!=2) rr_sync();
  }
}

typedef CUresult (*RR_Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static int rr_map_host(CUtensorMap* tm,int64_t p,int inner,int outer,int box) {
  static RR_Encode enc=nullptr;
  if(!enc) { cudaDriverEntryPointQueryResult q; if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&enc,12000,cudaEnableDefault,&q)!=cudaSuccess||!enc) return -100; }
  cuuint64_t dims[2]={(uint64_t)inner,(uint64_t)outer},stride[1]={(uint64_t)inner*2};
  cuuint32_t sz[2]={64,(uint32_t)box},es[2]={1,1};
  return (int)enc(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,sz,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
                 CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template<int BN,int S,int ST,int SW,int RED,int PIPE>
int rr_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int bytes=ST*(2*NR+TM)*128+ST*8*((PIPE&2)?2:1)+48+(RED>=1?64*BN*4:0);
  auto fn=rr_kernel<BN,S,ST,SW,RED,PIPE>;
  static bool configured=false;
  if(!configured) {
    cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(e!=cudaSuccess) return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
    if(e!=cudaSuccess)return -(int)e;
    configured=true;
  }
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(256/NR*S,(m+TM-1)/TM,1);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim={S,1,1};
  if(S>1){cfg.attrs=attr;cfg.numAttrs=1;}
  if(query==1){int n=0;cudaError_t e=cudaOccupancyMaxActiveClusters(&n,fn,&cfg);return e==cudaSuccess?n:-(int)e;}
  CUtensorMap tw,tx;
  int e=rr_map_host(&tw,w,64,256*128*2,NR*2);if(e)return e;
  e=rr_map_host(&tx,x,8192,m,TM);if(e)return e;
  cudaError_t err=cudaLaunchKernelEx(&cfg,fn,tw,tx,(float*)out,m);
  return err==cudaSuccess?0:(int)err;
}
int rr_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
 switch(id){@@CASES@@ default:return -99;}
}
@@SCALAR@@
'''

ROUTER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);
int rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);
'''


def build_router(configs):
    src = ROUTER_CUDA.replace('@@MMA@@', '\n'.join(_mma(n) for n in sorted({c[0] for c in configs})))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
    cases = '\n'.join(f'case {i}: return rr_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
    src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', ROUTER_SCALAR + ROUTER_SMALL)
    name = 'router_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, ROUTER_CPP, cuda_sources=src, functions=['rr_launch','rr_scalar','rr_small'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)


def pack_router(w, nr):
    hi = w.to(torch.bfloat16)
    lo = (w-hi.float()).to(torch.bfloat16)
    return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)


def prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext':build_router(configs), 'configs':configs,'weight':w,
            'packed':{nr:pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}


def run_candidate(x, st, index):
    out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rr_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rr_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out


# Configuration: WGMMA N, K splits, stages, swap operands, reduction, pipeline flags.
# Reduction 0 pulls, 1 pushes with a final barrier, 2 relies on destination byte completion.
# Pipeline bits: 1 keeps one MMA group in flight, 2 adds a producer, 4 initializes reduction first.
ROUTER_CONFIGS = [(32,16,4,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]


def prepare_router(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {'enabled':False}
    st = prepare_candidates(weights, ROUTER_CONFIGS)
    st['small'] = {(16,8):pack_small_router(w,16,8)}
    st['enabled'] = True
    for m in (1,8,16,32,128,256):
        x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
        run_router(x,'router',256,st)
    torch.cuda.synchronize(w.device)
    return st


def run_router(x, layer, n, st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return run_candidate(x,st,-1)
    if m <= 16:
        return run_small_candidate(x,st)
    return run_candidate(x,st,0 if m <= 32 else 1 if m <= 128 else 2)


ROUTER_SCALAR = r'''
template<int M, int NT, int ST>
__global__ void rr_scalar_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int rr_scalar(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=rr_scalar_kernel<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
'''

ROUTER_SMALL = r'''
template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void rr_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+S*NI;
  const uint32_t rb=rr_smem(sm+(S+WK)*NI*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank==0 && tid==0) {rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  if(rank==0 && warp==0) __syncwarp();
  if(rank==0 && tid==0) {rr_expect(rb,(S-1)*NI*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  for(int v=tid;v<NI;v+=WK*32) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) rr_add(sum,part[k*NI+v]);
    if(rank==0)recv[v]=sum;
    else rr_push(rr_map(rr_smem(recv+rank*NI+v),0),sum,rr_map(rb,0));
  }
  if(rank==0) {
    __syncthreads();rr_wait(rb,0);
    for(int v=tid;v<NI;v+=WK*32) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int s=0;s<S;s++) rr_add(sum,recv[s*NI+v]);
      const int ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
      if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
      if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
    }
  }
}

template<int MT,int S,int WK>
int rr_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=rr_small_kernel<MT,S,WK>;
  constexpr int bytes=(S+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int rr_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return rr_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return rr_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return rr_small_t<4,S,W>(w,x,out,m,stream,query);}
RSM(8,8) RSM(16,8) RSM(16,4)
return -99;
}

'''


def pack_small_router(w, s=16, wk=8):
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    def units(v):
        v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
        return v.permute(0,1,3,2,4)
    data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
    ku=128//s//wk
    return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()


def run_small_candidate(x,st,s=16,wk=8):
    out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    rc=st['ext'].rr_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'router small launch failed: {rc}')
    return out



V9_SELECTED_ROUTES = []


# ---- qkv/qkv_kernels.py ----



def qkv_mma(n):
    regs = ",".join(f"%{i}" for i in range(n // 2))
    outs = ",".join(f'"+f"(a[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{
      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));
    }}'''


QKV_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define QD __device__ __forceinline__
QD uint32_t q_addr(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}
QD void q_init(uint32_t p,int n) {asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_expect(uint32_t p,int n) {asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_wait(uint32_t p,int phase) {
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(p),"r"(phase):"memory");
}
QD void q_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}
QD uint32_t q_map(uint32_t p,int rank) {uint32_t d;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(d):"r"(p),"r"(rank));return d;}
QD void q_push(uint32_t p,float4 v,uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
QD uint64_t q_desc(uint32_t p) {return uint64_t((p&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}
QD uint64_t q_time() {uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
QD void q_tma(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k),"r"(r),"r"(b),"l"(policy):"memory");
}
@@MMA@@
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
__global__ __launch_bounds__((PIPE&2)?256:128) void q_kernel(const __grid_constant__ CUtensorMap tw,
    const __grid_constant__ CUtensorMap tx,__nv_bfloat16* out,int m,unsigned long long* stamps) {
  constexpr int KT=(PIPE&32)?2:1, LR=(2304+G-1)/G, WB=NR*128, XB=TM*128, SB=KT*(WB+XB);
  constexpr int NV=NR*TM/4, OWN=(PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128;
  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, rank=blockIdx.x%S, group=blockIdx.x/S;
  const int tok=blockIdx.y*TM, row=group*2304/G, rows=(group+1)*2304/G-row;
  const int kb=(128/KT)*rank/S, nkb=(128/KT)*(rank+1)/S-kb, own=min(OWN,NV-rank*OWN);
  const uint32_t base=q_addr(sm), bars=base+ST*SB, empty=bars+ST*8, rb=base+RB;
  uint64_t begin=0, stream_end=0;
  if constexpr(PROFILE) begin=q_time();
  if constexpr(LR<NR) {
    for(int i=tid;i<ST*(NR-LR)*32;i+=blockDim.x) {
      int stage=i/((NR-LR)*32), j=i%((NR-LR)*32);
      ((uint32_t*)(sm+stage*SB+LR*128))[j]=0;
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
  }
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(px));
  auto load=[&](int i) {
    const int stage=i%ST;
    q_expect(bars+stage*8,KT*(LR*128+XB));
    if constexpr(KT==2)q_tma(base+stage*SB,&tw,0,(group*128+(kb+i)*KT)*NR,bars+stage*8,pw);
    else q_tma(base+stage*SB,&tw,(kb+i)*64,group*LR,bars+stage*8,pw);
#pragma unroll
    for(int q=0;q<KT;q++)q_tma(base+stage*SB+KT*WB+q*XB,&tx,((kb+i)*KT+q)*64,tok,bars+stage*8,px);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) {q_init(bars+i*8,1);if constexpr(PIPE&2)q_init(empty+i*8,4);}
    if constexpr(S>1) q_init(rb,1);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    if constexpr(S>1) {q_expect(rb,(S-1)*own*16);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
#pragma unroll
    for(int i=0;i<ST;i++) if(i<nkb)load(i);
  }
  __syncthreads();
  if constexpr(S>1 && RED!=3) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) for(int i=ST;i<nkb;i++) {q_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
      if constexpr(S>1) {
        if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
        if constexpr(RED!=2) q_sync();
      }
      return;
    }
  }
  float a[(TM/64)*(NR/2)];
#pragma unroll
  for(int h=0;h<TM/64;h++)
#pragma unroll
    for(int j=0;j<NR/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(a[h*(NR/2)+j]));
#pragma unroll 1
  for(int i=0;i<nkb;i++) {
    const int stage=i%ST;
    q_wait(bars+stage*8,(i/ST)&1);
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
#pragma unroll
    for(int k=0;k<4*KT;k++) {
      if constexpr(PIPE&8) {
        uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
        @@SWAP@@
      } else {
#pragma unroll
        for(int h=0;h<TM/64;h++) {
          uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+h*8192+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
          @@CALL@@
        }
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    int done=i-((PIPE&1)?1:0);
    if(done>=0 && done+ST<nkb) {
      if constexpr(PIPE&2) {if(lane==0)asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(empty+(done%ST)*8):"memory");}
      else if(tid==0)load(done+ST);
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  if constexpr(PROFILE) stream_end=q_time();
  auto emit=[&](int v,float4 z) {
    int t=v%128, j=(v/128)%(NR/8), h=(v/128)/(NR/8);
    int rr=h*64+(t/32)*16+(t%32)/4, cc=j*8+(t%4)*2;
    float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      int tr,wc;
      if constexpr(PIPE&8) {tr=tok+(v/128)*8+(t%4)*2+(c&1);wc=(t/32)*16+(t%32)/4+(c/2)*8;}
      else {tr=tok+rr+(c/2)*8;wc=cc+(c&1);}
      if(tr<m && wc<rows)out[tr*2304+row+wc]=__float2bfloat16_rn(vv[c]);
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++)emit((h*(NR/8)+j)*128+tid,make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]));
  } else {
    if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++) {
        int v=(h*(NR/8)+j)*128+tid,dst=v/OWN,local=v%OWN;
        float4 z=make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]);
        uint32_t p=base+RECV+(rank*OWN+local)*16;
        if(dst==rank)*(float4*)(sm+RECV+(rank*OWN+local)*16)=z;
        else q_push(q_map(p,dst),z,q_map(rb,dst));
      }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;":::"memory");else __syncthreads();
    q_wait(rb,0);
    for(int t=tid;t<own;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) {
        float4 z=*(float4*)(sm+RECV+(r*OWN+t)*16);
        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;
      }
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED!=2) q_sync();
  }
  if constexpr(PROFILE) if(lane==0) {
    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
    int idx=((blockIdx.y*gridDim.x+blockIdx.x)*4+tid/32)*4;
    stamps[idx]=smid;stamps[idx+1]=begin;stamps[idx+2]=stream_end;stamps[idx+3]=q_time();
  }
}
using QEncode=decltype(&cuTensorMapEncodeTiled);
int q_tensor_map(CUtensorMap* tm,int64_t p,int inner,int rows,int box) {
  static QEncode encode=nullptr;
  if(!encode){cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);if(!encode)return -100;}
  uint64_t dims[2]={(uint64_t)inner,(uint64_t)rows},stride[1]={(uint64_t)inner*2};uint32_t b[2]={64,(uint32_t)box},es[2]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
int q_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=(PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128;
  constexpr int KT=(PIPE&32)?2:1;
  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?S*OWN*16:0);
  auto fn=q_kernel<NR,G,TM,S,ST,PIPE,RED,PROFILE>;
  static bool ready=false;
  if(!ready){cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(e!=cudaSuccess)return -(int)e;ready=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(G*S,(m+TM-1)/TM);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={S,1,1};if(S>1){cfg.attrs=&attr;cfg.numAttrs=1;}
  if(query==1){int v=0;cudaError_t e=S>1?cudaOccupancyMaxActiveClusters(&v,fn,&cfg):cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,fn,cfg.blockDim.x,bytes);return e==cudaSuccess?v:-(int)e;}
  CUtensorMap tw,tx;int e=q_tensor_map(&tw,w,KT==2?64:8192,KT==2?G*128*NR:G*LR,KT==2?KT*NR:LR);if(e)return e;e=q_tensor_map(&tx,x,8192,m,TM);if(e)return e;
  return int(cudaLaunchKernelEx(&cfg,fn,tw,tx,(__nv_bfloat16*)out,m,(unsigned long long*)stats));
}
int q_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  switch(id){@@CASES@@ default:return -99;}
}
'''


def build_qkv(configs, profile=False):
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
    source = QKV_CUDA.replace("@@MMA@@", "\n".join(qkv_mma(n) for n in ns))
    source = source.replace("@@CALL@@", " ".join(f"if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);" for n in ns))
    source = source.replace("@@SWAP@@", " ".join(f"if constexpr(TM=={n})q_mma_{n}(a,dw,dx);" for n in ns))
    source = source.replace("@@CASES@@", "\n".join(
        f'case {i}: return q_launch_t<{",".join(map(str,c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);'
        for i,c in enumerate(configs)))
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint q_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
    return load_inline("qkv_"+hashlib.sha256(source.encode()).hexdigest()[:16],cpp,cuda_sources=source,
        functions=["q_launch"],no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=["-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","--ptxas-options=-v"])


def qkv_pack(w, groups, kt=1):
    rows = (2304 + groups - 1) // groups
    if kt==2:
        return w.view(groups, rows, 128, 64).permute(0,2,1,3).contiguous().reshape(-1,64)
    if 2304 % groups == 0:
        return w.contiguous()
    out = torch.zeros((groups, rows, 8192), dtype=torch.bfloat16, device=w.device)
    for g in range(groups):
        lo, hi = g * 2304 // groups, (g + 1) * 2304 // groups
        out[g, :hi-lo].copy_(w[lo:hi])
    return out.reshape(-1,8192)


def prepare_qkv_candidates(weights, configs, profile=False):
    w = weights["qkv_gate"]
    return {"ext": build_qkv(configs, profile), "configs": list(configs),
        "packed": {(g,kt):qkv_pack(w,g,kt) for g,kt in {(c[1],2 if c[5]&32 else 1) for c in configs}}, "device":w.device}


def run_qkv_candidate(x, st, index, stats=None, query=0):
    out = torch.empty((x.shape[0],2304), dtype=torch.bfloat16, device=x.device)
    cfg = st["configs"][index]
    rc = st["ext"].q_launch(index, st["packed"][cfg[1],2 if cfg[5]&32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(),
        x.shape[0],torch.cuda.current_stream().cuda_stream,query,stats.data_ptr() if stats is not None else 0)
    if query:
        return rc
    if rc:
        raise RuntimeError(f"qkv launch {cfg} failed: {rc}")
    return out


# ---- qkv_sm/qkv_sm_kernels.py ----


QKV_SM_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define QKV_SMD __device__ __forceinline__

QKV_SMD int4 qkv_sm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

QKV_SMD uint4 qkv_sm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

QKV_SMD uint32_t qkv_sm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

QKV_SMD uint32_t qkv_sm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

QKV_SMD void qkv_sm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
QKV_SMD void qkv_sm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = qkv_sm_prmt(T0, T1, cm);
  const uint32_t hhi = qkv_sm_prmt(T0, T1, cm >> 16);
  a[0] = qkv_sm_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = qkv_sm_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = qkv_sm_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = qkv_sm_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void qkv_sm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
QKV_SMD void qkv_sm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) qkv_sm_never(p);
}

QKV_SMD uint32_t qkv_sm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

QKV_SMD uint32_t qkv_sm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

QKV_SMD uint32_t qkv_sm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

QKV_SMD void qkv_sm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

QKV_SMD void qkv_sm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

QKV_SMD void qkv_sm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
QKV_SMD void qkv_sm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
QKV_SMD void qkv_sm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

QKV_SMD void qkv_sm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

QKV_SMD bool qkv_sm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

QKV_SMD uint64_t qkv_sm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
QKV_SMD void qkv_sm_wait_bar(uint32_t bar) {
  if (qkv_sm_mbar_try(bar)) return;
  const uint64_t t0 = qkv_sm_gtime();
  while (!qkv_sm_mbar_try(bar))
    if (qkv_sm_gtime() - t0 > 20000000ull) break;
}

QKV_SMD void qkv_sm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

QKV_SMD float4 qkv_sm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

QKV_SMD void qkv_sm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
QKV_SMD void qkv_sm_store(float* p, float v) { *p = v; }

template <int FMT> struct QKV_SMFmt;
template <> struct QKV_SMFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct QKV_SMFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct QKV_SMFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct QKV_SMFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
QKV_SMD void qkv_sm_decode(const int4 (&w)[QKV_SMFmt<FMT>::NL], uint32_t (&A)[QKV_SMFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < QKV_SMFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < QKV_SMFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      qkv_sm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      qkv_sm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      qkv_sm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      qkv_sm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
QKV_SMD void qkv_sm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
QKV_SMD void qkv_sm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) qkv_sm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
__global__ void __launch_bounds__(NTMAX) qkv_sm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = QKV_SMFmt<FMT>::NL, NA = QKV_SMFmt<FMT>::NA, MTT = MT * TG;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
  const int wbytes = XM == 4 ? CWR * CWK * RT * NA * 2048 : 0;
  int4* sw = reinterpret_cast<int4*>(smem + xbytes);
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;
    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MTT, (m + 1) >> 1);
    const uint32_t bar = qkv_sm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = qkv_sm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        qkv_sm_mbar_init(bar, 1);
        qkv_sm_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        qkv_sm_fence_init();
      }
      qkv_sm_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = qkv_sm_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = qkv_sm_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = qkv_sm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if (XM != 4 || tg == 0) w[slot][j][l] = qkv_sm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = qkv_sm_ldx(xp);
          xv[slot][n][1] = qkv_sm_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    qkv_sm_order_point(m, outp);
    if constexpr (XM == 3 || XM == 5) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          qkv_sm_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
          if constexpr (XM == 7 || XM == 8 || XM == 11) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=qkv_sm_ldx(xp), z1=qkv_sm_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            qkv_sm_order_point(m, outp);
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) qkv_sm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          qkv_sm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) qkv_sm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
          if constexpr (XM == 4) {
            if (tg == 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q)
                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =
                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);
            }
            __syncthreads();
            if (tg != 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q) {
                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];
                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;
                }
            }
          }
          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  qkv_sm_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = qkv_sm_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  qkv_sm_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) qkv_sm_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
                }
              }
            }
          } else {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              uint32_t bx[8];
              if constexpr (XM >= 1 && XM <= 4 && !NOX) {
                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);
                const __nv_bfloat16* xp;
                if constexpr (XM == 2 || XM == 3 || XM == 5)
                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;
                else xp = x0 + (size_t)tok * K + si * 64;
                uint4 z0, z1;
                if constexpr (XM == 2 || XM == 3 || XM == 5) {
                  z0 = *reinterpret_cast<const uint4*>(xp);
                  z1 = *reinterpret_cast<const uint4*>(xp + 8);
                } else {
                  z0 = qkv_sm_ldx(xp); z1 = qkv_sm_ldx(xp + (XM >= 9 ? 32 : 8));
                }
                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;
                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;
              }
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);
                } else {
                  qkv_sm_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) qkv_sm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          qkv_sm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) qkv_sm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                qkv_sm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) qkv_sm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) qkv_sm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          const int NI = WR * RT * MTT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            qkv_sm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            qkv_sm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) qkv_sm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2 || RED == 4) {
            qkv_sm_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = qkv_sm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = (tg * MT + n) * 4 + t;
                  if (RED == 4 || tn < TV) {
                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else qkv_sm_st_async4(qkv_sm_mapa(qkv_sm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            qkv_sm_wait_bar(bar);
            if constexpr (RED == 4) qkv_sm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else qkv_sm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (RED == 4 || t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if ((tg * MT + n) * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            qkv_sm_cl_wait();
            const uint32_t rbar = qkv_sm_mapa(bar, 0);
            qkv_sm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else qkv_sm_st_async4(qkv_sm_mapa(qkv_sm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            qkv_sm_wait_bar(bar);
            qkv_sm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int qkv_sm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
static int qkv_sm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = qkv_sm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * TG * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT * TG, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;
    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;
  if (XM == 4) sm += (size_t)WR * WK * RT * QKV_SMFmt<FMT>::NA * 2048;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = qkv_sm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int qkv_sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
"""

QKV_SM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int qkv_sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

QKV_SM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
QKV_SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
QKV_SM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
QKV_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
QKV_SM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
QKV_SM_ESC_CAP = 32


def qkv_sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)


def qkv_sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,)))
        cases.append(f"    case {i}: return qkv_sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return QKV_SM_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def qkv_sm_build(instances, name="qkv_sm_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, QKV_SM_CPP + extra_cpp, cuda_sources=qkv_sm_cuda_source(instances) + extra_cuda,
                       functions=["qkv_sm_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def qkv_sm_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def qkv_sm_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def qkv_sm_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def qkv_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = qkv_sm_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def qkv_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = qkv_sm_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def qkv_sm_c12_unpack(data, base, esc):
    """Inverse of qkv_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def qkv_sm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def qkv_sm_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [qkv_sm_units(hi), qkv_sm_units(lo)]
    else:
        mats = [qkv_sm_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([qkv_sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = qkv_sm_c12_bytes(v)
        if not torch.equal(qkv_sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(qkv_sm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def qkv_sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def qkv_sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = qkv_sm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, QKV_SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > QKV_SM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = qkv_sm_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.qkv_sm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


# H100-selected candidates; H200 confirmation is recorded separately.
QKV_SM_TABLE_M = {}
QKV_SM_MAX_M = 32
QKV_SM_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def qkv_sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


def qkv_sm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def qkv_sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def qkv_sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if qkv_sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def qkv_sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def qkv_sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def qkv_sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = qkv_sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = qkv_sm_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = qkv_sm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = qkv_sm_verify_onehot(ext, inst_id[inst], sp, qkv_sm_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            qkv_sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


def prepare_qkv_sm(weights, table_m=None, table_mt=None, name="qkv_sm_gemv"):
    table_m = dict(QKV_SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(QKV_SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = qkv_sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, QKV_SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if qkv_sm_bf16_twin(inst):
            instances.append(qkv_sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = qkv_sm_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and qkv_sm_bf16_twin(inst):
            plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, qkv_sm_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    return st


def run_qkv_sm(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= QKV_SM_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", qkv_sm_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != QKV_SM_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if qkv_sm_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out



# H100 finalists. H200 routing stays disabled until the dated judge gate is used.
QKV_LARGE_CONFIGS = [(64,36,128,3,8,27,2), (64,36,128,3,8,25,2), (64,36,128,3,4,59,2)]
QKV_SMALL_INST = (1,2,1,1,4,0,256,0,2304,8192,1,8,8,2,7,1)
QKV_SMALL_GEOMETRY = (1,8,8,2)


def prepare_qkv(weights) -> dict:
    w = weights.get("qkv_gate")
    if w is None or w.shape != (2304,8192) or w.dtype != torch.bfloat16 or not w.is_cuda:
        return {"is_enabled":False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {"is_enabled":False}
    small = prepare_qkv_sm({"qkv_gate":w}, table_m={},
        table_mt={("qkv_gate",2):(QKV_SMALL_INST,QKV_SMALL_GEOMETRY)}, name="qkv_small_final")
    large = prepare_qkv_candidates(weights,QKV_LARGE_CONFIGS)
    warm = torch.zeros((128,8192),dtype=torch.bfloat16,device=w.device)
    for index in range(len(QKV_LARGE_CONFIGS)):
        run_qkv_candidate(warm,large,index)
    torch.cuda.synchronize(w.device)
    return {"is_enabled":"H100" in torch.cuda.get_device_name(w.device), "device":w.device,
        "small":small, "large":large, "large_index":0}


def run_qkv(x, layer, n, st) -> torch.Tensor | None:
    if not st.get("is_enabled") or layer != "qkv_gate" or n != 2304:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda:
        return None
    if x.device != st["device"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if 9 <= m <= 16:
        return run_qkv_sm(x,layer,n,st["small"])
    if m == 128:
        return run_qkv_candidate(x,st["large"],st["large_index"])
    return None

# ---- router2/router2_kernels.py ----



def rt2_mma(n):
    regs = ', '.join(f'%{i}' for i in range(n // 2))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void rt2_mma_{n}(float* d, uint64_t a, uint64_t b) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
}}'''


rt2_ROUTER_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define rt2_RD __device__ __forceinline__
rt2_RD uint32_t rt2_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
rt2_RD void rt2_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
rt2_RD void rt2_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
rt2_RD void rt2_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
rt2_RD void rt2_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
rt2_RD uint32_t rt2_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
rt2_RD void rt2_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
rt2_RD float4 rt2_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
rt2_RD void rt2_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
rt2_RD uint64_t rt2_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
rt2_RD void rt2_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}
@@MMA@@

template<int BN,int S,int ST,int SW,int RED,int PIPE>
__global__ __launch_bounds__((PIPE&2)?256:128) void rt2_kernel(const __grid_constant__ CUtensorMap tw,
                                               const __grid_constant__ CUtensorMap tx, float* out, int m) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int WB=NR*128, XB=TM*128, SB=WB*2+XB;
  constexpr int NF=64*BN, NV=NF/4, OWN=NV/S;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S;
  const int row=blockIdx.x/S*NR, tok=blockIdx.y*TM;
  const uint32_t base=rt2_smem(sm), bars=base+ST*SB, empty=bars+ST*8, rb=(empty+((PIPE&2)?ST*8:0)+15)&~15u;
  const uint32_t recv=rb+16;
  const int kb=rank*(128/S);
  auto load=[&](int i) {
    const int st=i%ST;
    const uint32_t b=bars+st*8, d=base+st*SB;
    rt2_expect(b,SB);
    rt2_tma(d,&tw,0,((row/NR)*128+kb+i)*NR*2,b);
    rt2_tma(d+2*WB,&tx,(kb+i)*64,tok,b);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) { rt2_init(bars+i*8,1); if constexpr(PIPE&2) rt2_init(empty+i*8,4); }
    asm volatile("fence.proxy.async.shared::cta;" :::"memory");
    if constexpr(PIPE&4) {
    if constexpr(RED>=1) {
      rt2_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rt2_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
#pragma unroll
    for(int i=0;i<ST && i<128/S;i++) load(i);
    if constexpr(!(PIPE&4)) {
    if constexpr(RED>=1) {
      rt2_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rt2_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
  }
  __syncthreads();
  if constexpr(RED>=1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) {
        for(int i=ST;i<128/S;i++) {
          rt2_wait(empty+(i%ST)*8,((i/ST)-1)&1);
          load(i);
        }
      }
      if constexpr(RED>=1) asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
      if constexpr(RED!=2) rt2_sync();
      return;
    }
  }
  float acc[BN/2];
#pragma unroll
  for(int j=0;j<BN/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(acc[j]));
  int prev=0;
#pragma unroll 1
  for(int i=0;i<128/S;i++) {
    const int st=i%ST;
    rt2_wait(bars+st*8,(i/ST)&1);
    const uint32_t w=base+st*SB, x=w+2*WB;
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;" :::"memory");
#pragma unroll
    for(int q=0;q<4;q++) {
      const uint64_t dx=rt2_desc(x+q*32),dh=rt2_desc(w+q*32),dl=rt2_desc(w+WB+q*32);
      if constexpr(SW) { @@CALL_H@@; @@CALL_L@@; }
      else { @@CALL_NH@@; @@CALL_NL@@; }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" :::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" :::"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    if constexpr(PIPE&2) {
      if constexpr(PIPE&1) {
        if(i>0 && prev+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+(prev%ST)*8):"memory");
      } else {
        if(i+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+st*8):"memory");
      }
    } else if constexpr(PIPE&1) {
      if(i>0 && prev+ST<128/S && tid==0) load(prev+ST);
    } else {
      if(i+ST<128/S && tid==0) load(i+ST);
    }
    prev=i;
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
  auto emit=[&](int v,float4 z) {
    const int t=v%128, j=v/128, ln=t&31, wr=t>>5;
    const int ar=wr*16+(ln>>2),bc=j*8+(ln&3)*2;
    const float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      const int a=ar+(c/2)*8, b=bc+(c&1);
      const int tr=tok+(SW?b:a), wc=row+(SW?a:b);
      if(tr<m) out[tr*256+wc]=vv[c];
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int j=0;j<BN/8;j++) emit(j*128+tid,make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]));
  } else if constexpr(RED==0) {
    float4* part=(float4*)sm;
#pragma unroll
    for(int j=0;j<BN/8;j++) part[j*128+tid]=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
    rt2_sync();
    for(int t=tid;t<OWN;t+=128) {
      const int v=rank*OWN+t;
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rt2_add(sum,rt2_pull(rt2_map(base+v*16,r)));
      emit(v,sum);
    }
    rt2_sync();
  } else {
    asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/8;j++) {
      const int v=j*128+tid, dst=v/OWN, local=v%OWN;
      const float4 val=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
      const uint32_t addr=recv+(rank*OWN+local)*16;
      if(dst==rank) *(float4*)(sm+(addr-base))=val;
      else rt2_push(rt2_map(addr,dst),val,rt2_map(rb,dst));
    }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;" :::"memory"); else __syncthreads();
    rt2_wait(rb,0);
    for(int t=tid;t<OWN;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rt2_add(sum,*(float4*)(sm+recv-base+(r*OWN+t)*16));
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED!=2) rt2_sync();
  }
}

typedef CUresult (*rt2_Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static int rt2_map_host(CUtensorMap* tm,int64_t p,int inner,int outer,int box,int promo) {
  static rt2_Encode enc=nullptr;
  if(!enc) { cudaDriverEntryPointQueryResult q; if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&enc,12000,cudaEnableDefault,&q)!=cudaSuccess||!enc) return -100; }
  cuuint64_t dims[2]={(uint64_t)inner,(uint64_t)outer},stride[1]={(uint64_t)inner*2};
  cuuint32_t sz[2]={64,(uint32_t)box},es[2]={1,1};
  return (int)enc(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,sz,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
                 CU_TENSOR_MAP_SWIZZLE_128B,(promo==64?CU_TENSOR_MAP_L2_PROMOTION_L2_64B:promo==128?CU_TENSOR_MAP_L2_PROMOTION_L2_128B:promo==0?CU_TENSOR_MAP_L2_PROMOTION_NONE:CU_TENSOR_MAP_L2_PROMOTION_L2_256B),CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template<int BN,int S,int ST,int SW,int RED,int PIPE,int WP=256,int XP=256>
int rt2_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int bytes=ST*(2*NR+TM)*128+ST*8*((PIPE&2)?2:1)+48+(RED>=1?64*BN*4:0);
  auto fn=rt2_kernel<BN,S,ST,SW,RED,PIPE>;
  static bool configured=false;
  if(!configured) {
    cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(e!=cudaSuccess) return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
    if(e!=cudaSuccess)return -(int)e;
    configured=true;
  }
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(256/NR*S,(m+TM-1)/TM,1);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim={S,1,1};
  if(S>1){cfg.attrs=attr;cfg.numAttrs=1;}
  if(query==1){int n=0;cudaError_t e=cudaOccupancyMaxActiveClusters(&n,fn,&cfg);return e==cudaSuccess?n:-(int)e;}
  CUtensorMap tw,tx;
  int e=rt2_map_host(&tw,w,64,256*128*2,NR*2,WP);if(e)return e;
  e=rt2_map_host(&tx,x,8192,m,TM,XP);if(e)return e;
  cudaError_t err=cudaLaunchKernelEx(&cfg,fn,tw,tx,(float*)out,m);
  return err==cudaSuccess?0:(int)err;
}
int rt2_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
 switch(id){@@CASES@@ default:return -99;}
}
@@SCALAR@@
'''

rt2_ROUTER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int rt2_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int rt2_scalar(int64_t,int64_t,int64_t,int,int64_t,int);
int rt2_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);
'''


def rt2_build_router(configs):
    src = rt2_ROUTER_CUDA.replace('@@MMA@@', '\n'.join(rt2_mma(n) for n in sorted({c[0] for c in configs})))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rt2_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
    cases = '\n'.join(f'case {i}: return rt2_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
    src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', rt2_ROUTER_SCALAR + rt2_ROUTER_SMALL)
    name = 'router2_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, rt2_ROUTER_CPP, cuda_sources=src, functions=['rt2_launch','rt2_scalar','rt2_small'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)


def rt2_pack_router(w, nr):
    hi = w.to(torch.bfloat16)
    lo = (w-hi.float()).to(torch.bfloat16)
    return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)


def rt2_prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext':rt2_build_router(configs), 'configs':configs,'weight':w,
            'packed':{nr:rt2_pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}


def rt2_run_candidate(x, st, index):
    out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rt2_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rt2_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out


# Configuration: WGMMA N, K splits, stages, swap, reduction, pipeline, W/x L2 promotion.
# Reduction 0 pulls, 1 pushes with a final barrier, 2 relies on destination byte completion.
# Pipeline bits: 1 keeps one MMA group in flight, 2 adds a producer, 4 initializes reduction first.
rt2_ROUTER_CONFIGS = [(32,16,4,1,2,1,256,256), (32,8,4,0,2,1,256,256),
                      (64,8,6,1,2,1,64,64), (64,4,8,1,2,3,256,256)]


def prepare_router2(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {'enabled':False}
    st = rt2_prepare_candidates(weights, rt2_ROUTER_CONFIGS)
    st['small'] = {(16,8):rt2_pack_small_router(w,16,8)}
    st['enabled'] = True
    for m in (1,8,16,32,64,128,256):
        x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
        run_router2(x,'router',256,st)
    torch.cuda.synchronize(w.device)
    return st


def run_router2(x, layer, n, st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return rt2_run_candidate(x,st,-1)
    if m <= 16:
        return rt2_run_small_candidate(x,st)
    return rt2_run_candidate(x,st,0 if m <= 32 else 1 if m <= 64 else 2 if m <= 128 else 3)


rt2_ROUTER_SCALAR = r'''
template<int M, int NT, int ST>
__global__ void rt2_scalar_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int rt2_scalar(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=rt2_scalar_kernel<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
'''

rt2_ROUTER_SMALL = r'''
template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void rt2_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+S*NI;
  const uint32_t rb=rt2_smem(sm+(S+WK)*NI*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank==0 && tid==0) {rt2_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  if(rank==0 && warp==0) __syncwarp();
  if(rank==0 && tid==0) {rt2_expect(rb,(S-1)*NI*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  for(int v=tid;v<NI;v+=WK*32) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) rt2_add(sum,part[k*NI+v]);
    if(rank==0)recv[v]=sum;
    else rt2_push(rt2_map(rt2_smem(recv+rank*NI+v),0),sum,rt2_map(rb,0));
  }
  if(rank==0) {
    __syncthreads();rt2_wait(rb,0);
    for(int v=tid;v<NI;v+=WK*32) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int s=0;s<S;s++) rt2_add(sum,recv[s*NI+v]);
      const int ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
      if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
      if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
    }
  }
}

template<int MT,int S,int WK>
int rt2_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=rt2_small_kernel<MT,S,WK>;
  constexpr int bytes=(S+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int rt2_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#define rt2_RSM(S,W) if(s==S && wk==W) {if(m<=8)return rt2_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return rt2_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return rt2_small_t<4,S,W>(w,x,out,m,stream,query);}
rt2_RSM(8,8) rt2_RSM(16,8) rt2_RSM(16,4)
return -99;
}

'''


def rt2_pack_small_router(w, s=16, wk=8):
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    def units(v):
        v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
        return v.permute(0,1,3,2,4)
    data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
    ku=128//s//wk
    return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()


def rt2_run_small_candidate(x,st,s=16,wk=8):
    out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    rc=st['ext'].rt2_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'router small launch failed: {rc}')
    return out

# ---- olayer/olayer_kernels.py ----




OLAYER_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");
}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
@@MMA@@

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        @@CALL@@
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);
      uint32_t d=base+2*(cb*72+rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for(int q=tid;q<T*(R/8);q+=TH) {
    int token=q/(R/8),row=(q%(R/8))*8;
    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);
  }
}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {@@CASES@@ default:return -99;}
}
'''

OLAYER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
'''


def olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join(f'%{i}' for i in range(n // 2))
        outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
        functions.append(f'''OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{
          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));
        }}''')
    cases = '\n'.join(f'case {i}: return ol_launch_t<{",".join(map(str,tuple(c)+(1,) if len(c)==6 else c))}>(x,w,out,m,stream,query);' for i,c in enumerate(configs))
    calls = '\n'.join(f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns)
    return OLAYER_CUDA.replace('@@MMA@@','\n'.join(functions)).replace('@@CASES@@',cases).replace('@@CALL@@',calls)


def build_olayer(configs):
    for cfg in configs:
        bn,bk,st,pipe,pack,hint,*groups = cfg
        tw = groups[0] if groups else 1
        if (bn<16 or bn>256 or bn%8 or bk not in (64,128,256) or st<1
                or st*((64+bn*tw)*bk*2+16)>227000 or pipe not in (0,1,2,3)
                or (pipe&1 and st<2) or pack not in (0,1,2) or hint not in (0,1)
                or tw not in (1,2) or bn*tw>256 or (tw>1 and not pipe&2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = olayer_source(configs)
    name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name,OLAYER_CPP,cuda_sources=source,functions=['ol_launch'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True,with_pytorch_error_handling=False)


def pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64,device=weight.device)[None,:] ^ ((torch.arange(64,device=weight.device)[:,None]&7)*8)
    tiled = weight.reshape(128,64,16,64).permute(0,2,1,3)
    return tiled.gather(-1,columns.expand(128,16,64,64)).contiguous()


# Token tile, K tile, stages, pipeline flags, packed weight, L2 hints.
OLAYER_PRIMARY_CONFIGS = [(16,256,3,3,1,1),(32,128,8,1,1,1),(64,128,4,3,1,1),
                         (128,128,4,3,1,1),(128,64,5,3,1,1,2)]
OLAYER_NEARBY_CONFIGS = {
    bn: (bn,128,4,3,1,1) if bn<=128 else (bn,64,min(8,227000//((64+bn)*128+16)),3,1,1)
    for bn in range(72,249,8)
}
OLAYER_CONFIGS = list(dict.fromkeys(OLAYER_PRIMARY_CONFIGS+list(OLAYER_NEARBY_CONFIGS.values())))


def prepare_olayer_candidates(weights,configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled':True,'ext':build_olayer(configs),'configs':configs,'weight':weight,
            'packed':pack_olayer(weight) if any(c[4]==1 for c in configs) else None,
            'tiled32':weight.reshape(256,32,16,64).permute(0,2,1,3).contiguous() if any(c[4]==2 for c in configs) else None}


def run_olayer_candidate(x,st,index) -> torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack==2 else st['packed'] if pack==1 else st['weight']
    out = torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
    rc = st['ext'].ol_launch(index,x.data_ptr(),weight.data_ptr(),out.data_ptr(),x.shape[0],torch.cuda.current_stream().cuda_stream,0)
    if rc:
        raise RuntimeError(f'olayer launch failed: {rc}, config={st["configs"][index]}')
    return out


def prepare_olayer(weights) -> dict:
    weight = weights.get('o')
    if weight is None or weight.shape != (8192,1024) or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(weight.device) != (9,0):
        return {'enabled':False}
    st = prepare_olayer_candidates(weights,OLAYER_CONFIGS)
    st['token_configs'] = {bn:OLAYER_CONFIGS.index(cfg) for bn,cfg in OLAYER_NEARBY_CONFIGS.items()}
    for index,cfg in enumerate(OLAYER_CONFIGS):
        m = cfg[0]*(cfg[6] if len(cfg)>6 else 1)
        run_olayer_candidate(torch.zeros((m,1024),dtype=torch.bfloat16,device=weight.device),st,index)
    torch.cuda.synchronize(weight.device)
    return st


def run_olayer(x,layer,n,st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'o' or n != 8192:
        return None
    if x.ndim != 2 or x.shape[1] != 1024 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr()%16 or not 1 <= x.shape[0] <= 256:
        return None
    m = x.shape[0]
    if 65<=m<=224:
        index = st['token_configs'][((m+7)//8)*8]
        return run_olayer_candidate(x,st,index)
    return None

# ---- cpstage/cpstage_kernels.py ----


CPS32_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define M32D __device__ __forceinline__

M32D int4 cps32_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint4 cps32_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint32_t cps32_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

M32D uint32_t cps32_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

M32D void cps32_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
M32D void cps32_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = cps32_prmt(T0, T1, cm);
  const uint32_t hhi = cps32_prmt(T0, T1, cm >> 16);
  a[0] = cps32_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = cps32_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = cps32_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = cps32_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void cps32_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
M32D void cps32_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) cps32_never(p);
}

M32D uint32_t cps32_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

M32D uint32_t cps32_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

M32D uint32_t cps32_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

M32D void cps32_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

M32D void cps32_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

M32D void cps32_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
M32D void cps32_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
M32D void cps32_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

M32D void cps32_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

M32D bool cps32_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

M32D uint64_t cps32_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
M32D void cps32_wait_bar(uint32_t bar) {
  if (cps32_mbar_try(bar)) return;
  const uint64_t t0 = cps32_gtime();
  while (!cps32_mbar_try(bar))
    if (cps32_gtime() - t0 > 20000000ull) break;
}

M32D void cps32_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

M32D float4 cps32_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

M32D void cps32_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
M32D void cps32_store(float* p, float v) { *p = v; }

template <int FMT> struct M32Fmt;
template <> struct M32Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct M32Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct M32Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct M32Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
M32D void cps32_decode(const int4 (&w)[M32Fmt<FMT>::NL], uint32_t (&A)[M32Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < M32Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < M32Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      cps32_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      cps32_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      cps32_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      cps32_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
M32D void cps32_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
M32D void cps32_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) cps32_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>
__global__ void __launch_bounds__(NTMAX) cps32_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = M32Fmt<FMT>::NL, NA = M32Fmt<FMT>::NA, MTT = MT * TG;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  const int wbytes = rawbytes + (XM == 4 ? CWR * CWK * RT * NA * 2048 : 0);
  int4* sw = reinterpret_cast<int4*>(smem + xbytes);
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;
    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MTT, (m + 1) >> 1);
    const uint32_t bar = cps32_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = cps32_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        cps32_mbar_init(bar, 1);
        cps32_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        cps32_fence_init();
      }
      cps32_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = cps32_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = cps32_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = cps32_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4* rawring = reinterpret_cast<int4*>(smem + xbytes);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=cps32_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) if (XM != 4 || tg == 0) w[slot][j][l] = cps32_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = cps32_ldx(xp);
          xv[slot][n][1] = cps32_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    cps32_order_point(m, outp);
    if constexpr (XM == 3 || XM == 5) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          cps32_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
          if constexpr (XM == 7 || XM == 8 || XM == 11) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=cps32_ldx(xp), z1=cps32_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            cps32_order_point(m, outp);
          }
          if constexpr (CP > 0) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=cps32_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) cps32_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if constexpr (CP > 0) stage_weights(si%CP,si+CP);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          cps32_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) cps32_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
          if constexpr (XM == 4) {
            if (tg == 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q)
                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =
                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);
            }
            __syncthreads();
            if (tg != 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q) {
                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];
                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;
                }
            }
          }
          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  cps32_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = cps32_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  cps32_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) cps32_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
                }
              }
            }
          } else {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              uint32_t bx[8];
              if constexpr (XM >= 1 && XM <= 4 && !NOX) {
                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);
                const __nv_bfloat16* xp;
                if constexpr (XM == 2 || XM == 3 || XM == 5)
                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;
                else xp = x0 + (size_t)tok * K + si * 64;
                uint4 z0, z1;
                if constexpr (XM == 2 || XM == 3 || XM == 5) {
                  z0 = *reinterpret_cast<const uint4*>(xp);
                  z1 = *reinterpret_cast<const uint4*>(xp + 8);
                } else {
                  z0 = cps32_ldx(xp); z1 = cps32_ldx(xp + (XM >= 9 ? 32 : 8));
                }
                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;
                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;
              }
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);
                } else {
                  cps32_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) cps32_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          cps32_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) cps32_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                cps32_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) cps32_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) cps32_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          const int NI = WR * RT * MTT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            cps32_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            cps32_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) cps32_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2 || RED == 4) {
            cps32_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = cps32_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = (tg * MT + n) * 4 + t;
                  if (RED == 4 || tn < TV) {
                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else cps32_st_async4(cps32_mapa(cps32_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            cps32_wait_bar(bar);
            if constexpr (RED == 4) cps32_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else cps32_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (RED == 4 || t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if ((tg * MT + n) * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            cps32_cl_wait();
            const uint32_t rbar = cps32_mapa(bar, 0);
            cps32_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else cps32_st_async4(cps32_mapa(cps32_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            cps32_wait_bar(bar);
            cps32_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int cps32_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>
static int cps32_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = cps32_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * TG * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT * TG, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;
    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;
  if (XM == 4) sm += (size_t)WR * WK * RT * M32Fmt<FMT>::NA * 2048;
  sm += (size_t)CP * NW * RT * M32Fmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = cps32_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int cps32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
"""

CPS32_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int cps32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

CPS32_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")

CPS32_NL = {0: 4, 1: 3, 2: 8, 3: 6}

CPS32_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}

CPS32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

CPS32_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}

CPS32_ESC_CAP = 32

def cps32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def cps32_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def cps32_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cps32_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

def cps32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = cps32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def cps32_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = cps32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def cps32_c12_unpack(data, base, esc):
    """Inverse of cps32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def cps32_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def cps32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [cps32_units(hi), cps32_units(lo)]
    else:
        mats = [cps32_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([cps32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = cps32_c12_bytes(v)
        if not torch.equal(cps32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(cps32_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def cps32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def cps32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = cps32_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, CPS32_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > CPS32_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = cps32_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cps32_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

CPS32_TABLE_M = {}

CPS32_MAX_M = 32

CPS32_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cps32_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def cps32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cps32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if cps32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cps32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def cps32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def cps32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = cps32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = cps32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = cps32_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = cps32_verify_onehot(ext, inst_id[inst], sp, cps32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            cps32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)



CPSSM_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define CPSSD __device__ __forceinline__

CPSSD int4 cpssm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint4 cpssm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint32_t cpssm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

CPSSD uint32_t cpssm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

CPSSD void cpssm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
CPSSD void cpssm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = cpssm_prmt(T0, T1, cm);
  const uint32_t hhi = cpssm_prmt(T0, T1, cm >> 16);
  a[0] = cpssm_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = cpssm_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = cpssm_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = cpssm_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void cpssm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
CPSSD void cpssm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) cpssm_never(p);
}

CPSSD uint32_t cpssm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

CPSSD uint32_t cpssm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

CPSSD uint32_t cpssm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

CPSSD void cpssm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

CPSSD void cpssm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

CPSSD void cpssm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
CPSSD void cpssm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
CPSSD void cpssm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

CPSSD void cpssm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

CPSSD bool cpssm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

CPSSD uint64_t cpssm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
CPSSD void cpssm_wait_bar(uint32_t bar) {
  if (cpssm_mbar_try(bar)) return;
  const uint64_t t0 = cpssm_gtime();
  while (!cpssm_mbar_try(bar))
    if (cpssm_gtime() - t0 > 20000000ull) break;
}

CPSSD void cpssm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

CPSSD float4 cpssm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

CPSSD void cpssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
CPSSD void cpssm_store(float* p, float v) { *p = v; }

template <int FMT> struct CpsSmFmt;
template <> struct CpsSmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct CpsSmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct CpsSmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct CpsSmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
CPSSD void cpssm_decode(const int4 (&w)[CpsSmFmt<FMT>::NL], uint32_t (&A)[CpsSmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < CpsSmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < CpsSmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      cpssm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      cpssm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      cpssm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      cpssm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
CPSSD void cpssm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
CPSSD void cpssm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) cpssm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
__global__ void __launch_bounds__(NTMAX) cpssm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = CpsSmFmt<FMT>::NL, NA = CpsSmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = cpssm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = cpssm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        cpssm_mbar_init(bar, 1);
        cpssm_mbar_expect(bar, parts * RT * TV * 128u);
        cpssm_fence_init();
      }
      cpssm_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = cpssm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4* rawring = reinterpret_cast<int4*>(smem);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=cpssm_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = cpssm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = cpssm_ldx(xp);
          xv[slot][n][1] = cpssm_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    cpssm_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          cpssm_order_point(m, outp);
        } else {
          if constexpr (CP > 0) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=cpssm_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) cpssm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if constexpr (CP > 0) stage_weights(si%CP,si+CP);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          cpssm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) cpssm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  cpssm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) cpssm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          cpssm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) cpssm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                cpssm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + rawbytes);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            cpssm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            cpssm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) cpssm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + rawbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            cpssm_cl_wait();
            if (t < TV) {
              const uint32_t rbar = cpssm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else cpssm_st_async4(cpssm_mapa(cpssm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            cpssm_wait_bar(bar);
            cpssm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            cpssm_cl_wait();
            const uint32_t rbar = cpssm_mapa(bar, 0);
            cpssm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else cpssm_st_async4(cpssm_mapa(cpssm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            cpssm_wait_bar(bar);
            cpssm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int cpssm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
static int cpssm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = cpssm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  sm += (size_t)CP * NW * RT * CpsSmFmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = cpssm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
"""

CPSSM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

CPSSM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")

CPSSM_NL = {0: 4, 1: 3, 2: 8, 3: 6}

CPSSM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}

CPSSM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

CPSSM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}

CPSSM_ESC_CAP = 32

def cpssm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def cpssm_c12_unpack(data, base, esc):
    """Inverse of cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def cpssm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [cpssm_units(hi), cpssm_units(lo)]
    else:
        mats = [cpssm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = cpssm_c12_bytes(v)
        if not torch.equal(cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(cpssm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > CPSSM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = cpstage___to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cpssm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

CPSSM_TABLE_M = {
    ("router", 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8)),
}

CPSSM_TABLE_MT = {
    ("pooler", 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)),
    ("qkv_gate", 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)),
    ("o", 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("o", 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("unpooler", 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)),
    ("router", 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)),
}

CPSSM_MAX_M = 32

CPSSM_TABLE_M.update({
    ("router", 8): CPSSM_TABLE_MT[("router", 1)],
    ("qkv_gate", 8): CPSSM_TABLE_MT[("qkv_gate", 1)],
    ("pooler", 8): CPSSM_TABLE_MT[("pooler", 1)],
    ("o", 8): CPSSM_TABLE_MT[("o", 1)],
    ("unpooler", 1): CPSSM_TABLE_MT[("unpooler", 1)],
    ("unpooler", 8): CPSSM_TABLE_MT[("unpooler", 1)],
})

CPSSM_TABLE_MT[('pooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))

CPSSM_TABLE_MT[('pooler', 4)] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))

CPSSM_TABLE_MT[('qkv_gate', 2)] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))

CPSSM_TABLE_MT[('unpooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def cpssm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cpssm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = cpssm_verify_onehot(ext, inst_id[inst], sp, cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def cpstage___to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cpstage___to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')


def cps_build(family, instances, depth):
    is32 = family == 'm32'
    prefix = 'cps32_' if is32 else 'cpssm_'
    template = CPS32_CUDA_TEMPLATE if is32 else CPSSM_CUDA_TEMPLATE
    cpp = CPS32_CPP if is32 else CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == (16 if is32 else 14) else inst + (0, 1) if is32 and len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name, cpp, cuda_sources=source, functions=[prefix+'launch'],
                       extra_cuda_cflags=['-O3','-lineinfo','-Xptxas=-v','-gencode=arch=compute_90a,code=sm_90a'],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def prepare_cps(weights, depth=None, layers=CPS_LAYERS, buckets=(2,4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple(layer for layer in layers if layer in ('pooler','unpooler'))
        for bucket,selected_depth in ((2,2),(4,1)):
            if bucket in buckets and selected_layers:
                part = prepare_cps(weights,selected_depth,selected_layers,(bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0,1,2,3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm','m32'):
        table = CPSSM_TABLE_MT if family == 'sm' else CPS32_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and layer in CPS_LAYERS and bucket == 1 and inst[4] in (2,3)
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and (layer,bucket) in CPS32_TABLE_MT:
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0,rows,CPSSM_LAYER_K[layer],*cfg)
            entries.append(((layer,'mt',bucket),(inst,cfg)))
        if include_policy and family == 'sm':
            for (layer,m),(inst,cfg) in CPSSM_TABLE_M.items():
                if layer not in CPS_LAYERS or layer not in weights or inst[4] not in (2,3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m,rows,CPSSM_LAYER_K[layer],*cfg)
                entries.append(((layer,'m',m),(inst,cfg)))
        if not entries:
            continue
        twin = cpssm_bf16_twin if family == 'sm' else cps32_bf16_twin
        instances = list(dict.fromkeys(inst for _,(inst,_) in entries for inst in (inst,twin(inst)) if inst is not None))
        ext = cps_build(family,instances,depth)
        ids = {inst:i for i,inst in enumerate(instances)}
        plan_fn = cpssm_plan if family == 'sm' else cps32_plan
        packs, streams = {}, {}
        for key,(inst,cfg) in entries:
            sp = plan_fn(ext,ids,packs,streams,weights[key[0]],key[0],key,inst,cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext,ids,packs,streams,weights[key[0]],key[0],key,chosen,cfg)
            if sp is not None:
                st['plans'][key] = dict(sp,ext=ext,family=family,inst=chosen,depth=depth)
    return st


def cps_launch(sp, x, out, n, policy=0, query=0):
    fn = cps32_launch_raw if sp['family'] == 'm32' else cpssm_launch_raw
    return fn(sp['ext'],sp['inst_id'],sp,x,out,n,sp['cfg'],query+policy*100)


def run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer,'m',m)) or st['plans'].get((layer,'mt',cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or x.shape[1] != CPS32_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = torch.empty((m,n),dtype=torch.float32 if layer == 'router' else torch.bfloat16,device=x.device)
    rc = cps_launch(sp,x,out,n,st.get('policy',0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out


def cps_resources(st, layer, m, policy=0):
    sp=st['plans'].get((layer,'m',m)) or st['plans'][layer,'mt',cps32_mt(m)]
    x=torch.empty((m,CPS32_LAYER_K[layer]),dtype=torch.bfloat16,device=sp['wp'].device)
    out=torch.empty((m,sp['rows']),dtype=torch.float32 if layer=='router' else torch.bfloat16,device=x.device)
    attrs=cps_launch(sp,x,out,sp['rows'],policy,2)
    return {'registers':attrs//1000,'local_bytes':attrs%1000,
            'active_clusters_or_blocks':cps_launch(sp,x,out,sp['rows'],policy,1),
            'blocks_per_sm':cps_launch(sp,x,out,sp['rows'],policy,3),
            'dynamic_smem':cps_launch(sp,x,out,sp['rows'],policy,4),
            'fmt':sp['fmt'],'inst':sp['inst'],'cfg':sp['cfg']}



"""BF16 cuBLASLt projections with offline descriptors and a cold-L2 tuning probe."""
import statistics
import time
import torch
from torch.utils.cpp_extension import load_inline
tt_lt__LT_CPP = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>\n#include <cublasLt.h>\n#include <algorithm>\n#include <array>\n#include <memory>\n#include <set>\n#include <vector>\n\n#define LT_OK(call) TORCH_CHECK((call) == CUBLAS_STATUS_SUCCESS, "cuBLASLt failure: ", #call)\n\nusing Config = std::array<int64_t, 9>;\n\nConfig config_of(const cublasLtMatmulAlgo_t& a) {\n  Config c{};\n  for (int i = 0; i < 9; ++i) {\n    size_t written = 0;\n    if (i >= 7) {\n      uint16_t value = 0;\n      cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i,\n                                          &value, sizeof(value), &written);\n      c[i] = value;\n    } else {\n      uint32_t value = 0;\n      cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i,\n                                          &value, sizeof(value), &written);\n      c[i] = value;\n    }\n  }\n  return c;\n}\n\nbool set_config(cublasLtMatmulAlgo_t& a, int attr, int value) {\n  if (attr >= 7) {\n    uint16_t v = value;\n    return cublasLtMatmulAlgoConfigSetAttribute(&a,\n      (cublasLtMatmulAlgoConfigAttributes_t)attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;\n  }\n  uint32_t v = value;\n  return cublasLtMatmulAlgoConfigSetAttribute(&a,\n    (cublasLtMatmulAlgoConfigAttributes_t)attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;\n}\n\nstd::vector<uint32_t> cap_list(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {\n  size_t bytes = 0;\n  cublasLtMatmulAlgoCapGetAttribute(&a, attr, nullptr, 0, &bytes);\n  if (!bytes) return {0};\n  std::vector<uint32_t> values(bytes / sizeof(uint32_t));\n  if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, values.data(), bytes, &bytes) != CUBLAS_STATUS_SUCCESS)\n    return {0};\n  return values;\n}\n\nuint32_t cap_int(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {\n  uint32_t v = 0;\n  size_t written = 0;\n  cublasLtMatmulAlgoCapGetAttribute(&a, attr, &v, sizeof(v), &written);\n  return v;\n}\n\nstruct LtContext {\n  cublasLtHandle_t handle = nullptr;\n  torch::Tensor workspace;\n  explicit LtContext(torch::Tensor ws) : workspace(ws) {\n    c10::cuda::CUDAGuard guard(ws.device());\n    LT_OK(cublasLtCreate(&handle));\n  }\n  ~LtContext() { if (handle) cublasLtDestroy(handle); }\n};\n\nstruct LtPlan {\n  std::shared_ptr<LtContext> context;\n  cublasLtMatmulDesc_t operation = nullptr;\n  cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, d_layout = nullptr;\n  torch::Tensor weight, bias;\n  int m, n, k, layout, epilogue;\n  std::vector<cublasLtMatmulAlgo_t> algorithms;\n  std::vector<size_t> workspaces;\n  std::set<Config> seen;\n  std::vector<std::vector<int64_t>> diagnostics;\n  int checked = 0, heuristic_count = 0, id_count = 0;\n\n  LtPlan(std::shared_ptr<LtContext> ctx, torch::Tensor w, int rows, int cols,\n         int inner, int variant, int epi)\n      : context(ctx), weight(w), m(rows), n(cols), k(inner), layout(variant), epilogue(epi) {\n    c10::cuda::CUDAGuard guard(w.device());\n    LT_OK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));\n    int8_t fast = 0;\n    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));\n    bool is_row = layout % 2;\n    bool is_packed = layout >= 2;\n    cublasOperation_t trans_a = !is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;\n    cublasOperation_t trans_b = is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;\n    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));\n    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));\n    if (is_row) {\n      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, m, k, k));\n      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, is_packed ? k : n,\n                                      is_packed ? n : k, is_packed ? n : k));\n      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, m, n, n));\n      cublasLtOrder_t order = CUBLASLT_ORDER_ROW;\n      for (auto d : {a_layout, b_layout, d_layout})\n        LT_OK(cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));\n    } else {\n      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, is_packed ? n : k,\n                                      is_packed ? k : n, is_packed ? n : k));\n      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, k, m, k));\n      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, n, m, n));\n    }\n    cublasLtEpilogue_t e = epi ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;\n    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));\n    if (epi) {\n      bias = torch::zeros({std::max(m, n)}, w.options());\n      void* ptr = bias.data_ptr();\n      LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &ptr, sizeof(ptr)));\n    }\n  }\n\n  ~LtPlan() {\n    if (operation) cublasLtMatmulDescDestroy(operation);\n    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);\n    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);\n    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);\n  }\n\n  bool add(const cublasLtMatmulAlgo_t& algo) {\n    Config c = config_of(algo);\n    if (seen.count(c)) return false;\n    seen.insert(c);\n    ++checked;\n    if (diagnostics.size() < 32) {\n      uint64_t flags = 0;\n      size_t written = 0;\n      int status = cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,\n        &flags, sizeof(flags), &written);\n      std::vector<int64_t> row(c.begin(), c.end());\n      row.push_back(flags);\n      row.push_back(status);\n      diagnostics.push_back(row);\n    }\n    if (c[3] != CUBLASLT_REDUCTION_SCHEME_NONE && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)\n      return false;\n    if (c[2] > 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return false;\n    uint64_t flags = 0;\n    size_t written = 0;\n    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,\n                                         &flags, sizeof(flags), &written) != CUBLAS_STATUS_SUCCESS) return false;\n    if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F)\n      return false;\n    // Algo 66 reports INPUT_16F even for CUDA_R_16BF descriptors in cuBLASLt 13.\n    // The descriptors fix BF16 storage; reject TF32 and require FP32 accumulators.\n    if (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)\n      return false;\n    cublasLtMatmulHeuristicResult_t result{};\n    if (cublasLtMatmulAlgoCheck(context->handle, operation, a_layout, b_layout,\n                               d_layout, d_layout, &algo, &result) != CUBLAS_STATUS_SUCCESS) return false;\n    if (result.state != CUBLAS_STATUS_SUCCESS || result.workspaceSize > size_t(context->workspace.numel())) return false;\n    algorithms.push_back(algo);\n    workspaces.push_back(result.workspaceSize);\n    return true;\n  }\n\n  int restore(std::vector<int64_t> values) {\n    TORCH_CHECK(values.size() == 9, "expected nine cuBLASLt configuration fields");\n    cublasLtMatmulAlgo_t algo{};\n    if (cublasLtMatmulAlgoInit(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,\n        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, values[0], &algo) != CUBLAS_STATUS_SUCCESS) return -1;\n    for (int i = 1; i < 9; ++i) {\n      if (!set_config(algo, i, values[i]) && values[i] != 0) return -1;\n    }\n    if (!add(algo)) return -1;\n    return algorithms.size() - 1;\n  }\n\n  void enumerate(bool exhaustive) {\n    c10::cuda::CUDAGuard guard(weight.device());\n    cublasLtMatmulPreference_t pref;\n    LT_OK(cublasLtMatmulPreferenceCreate(&pref));\n    uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;\n    LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,\n                                             &reduction, sizeof(reduction)));\n    for (size_t ws : {size_t(0), size_t(4 << 20), size_t(32 << 20), size_t(128 << 20)}) {\n      ws = std::min(ws, size_t(context->workspace.numel()));\n      LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));\n      std::vector<cublasLtMatmulHeuristicResult_t> result(256);\n      int count = 0;\n      auto rc = cublasLtMatmulAlgoGetHeuristic(context->handle, operation, a_layout, b_layout,\n                                               d_layout, d_layout, pref, 256, result.data(), &count);\n      if (rc != CUBLAS_STATUS_SUCCESS) continue;\n      heuristic_count += count;\n      for (int i = 0; i < count; ++i) if (result[i].state == CUBLAS_STATUS_SUCCESS) add(result[i].algo);\n    }\n    cublasLtMatmulPreferenceDestroy(pref);\n    if (!exhaustive) return;\n    std::array<int, 256> ids{};\n    LT_OK(cublasLtMatmulAlgoGetIds(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,\n      CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids.size(), ids.data(), &id_count));\n    for (int i = 0; i < id_count; ++i) {\n      cublasLtMatmulAlgo_t a{};\n      if (cublasLtMatmulAlgoInit(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,\n          CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[i], &a) != CUBLAS_STATUS_SUCCESS) continue;\n      auto tiles = cap_list(a, CUBLASLT_ALGO_CAP_TILE_IDS);\n      auto stages = cap_list(a, CUBLASLT_ALGO_CAP_STAGES_IDS);\n      for (auto tile : tiles) for (auto stage : stages) {\n        auto b = a;\n        if (!set_config(b, 1, tile) || !set_config(b, 6, stage)) continue;\n        add(b);\n        if (tiles.size() > 100) {\n          for (int cluster : {2, 3, 4, 5, 6, 7, 8, 9, 11, 12}) {\n            auto s = b;\n            if (set_config(s, 8, cluster)) add(s);\n          }\n        }\n      }\n    }\n  }\n\n  void expand(std::vector<int> indices) {\n    for (int index : indices) {\n      auto a = algorithms.at(index);\n      if (cap_int(a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT) &&\n          (cap_int(a, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {\n        for (int split : {2, 3, 4, 6, 8, 12, 16}) {\n          auto b = a;\n          if (set_config(b, 2, split) && set_config(b, 3, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) add(b);\n        }\n      }\n      auto stages = cap_list(a, CUBLASLT_ALGO_CAP_STAGES_IDS);\n      for (auto stage : stages) {\n        auto b = a;\n        if (set_config(b, 6, stage)) add(b);\n      }\n      for (int swizzle = 0; swizzle <= int(cap_int(a, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT)); ++swizzle) {\n        for (int custom = 0; custom <= int(cap_int(a, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX)); ++custom) {\n          for (int cluster : {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12}) {\n            auto b = a;\n            if (set_config(b, 4, swizzle) && set_config(b, 5, custom) && set_config(b, 8, cluster)) add(b);\n          }\n        }\n      }\n    }\n  }\n\n  std::vector<std::vector<int64_t>> metadata() const {\n    std::vector<std::vector<int64_t>> result;\n    for (size_t i = 0; i < algorithms.size(); ++i) {\n      Config c = config_of(algorithms[i]);\n      std::vector<int64_t> row(c.begin(), c.end());\n      row.push_back(workspaces[i]);\n      result.push_back(row);\n    }\n    return result;\n  }\n\n  int run(torch::Tensor x, torch::Tensor out, int index) {\n    const float alpha = 1.0f, beta = 0.0f;\n    const void* a = layout % 2 ? x.data_ptr() : weight.data_ptr();\n    const void* b = layout % 2 ? weight.data_ptr() : x.data_ptr();\n    return int(cublasLtMatmul(context->handle, operation, &alpha, a, a_layout, b, b_layout,\n      &beta, out.data_ptr(), d_layout, out.data_ptr(), d_layout, &algorithms.at(index),\n      context->workspace.data_ptr(), workspaces.at(index), at::cuda::getCurrentCUDAStream(x.get_device())));\n  }\n};\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def("version", []() { return cublasLtGetVersion(); });\n  pybind11::class_<LtContext, std::shared_ptr<LtContext>>(m, "Context")\n    .def(pybind11::init<torch::Tensor>());\n  pybind11::class_<LtPlan, std::shared_ptr<LtPlan>>(m, "Plan")\n    .def(pybind11::init<std::shared_ptr<LtContext>, torch::Tensor, int, int, int, int, int>())\n    .def("enumerate", &LtPlan::enumerate).def("expand", &LtPlan::expand)\n    .def("metadata", &LtPlan::metadata).def("restore", &LtPlan::restore).def("run", &LtPlan::run)\n    .def_readonly("checked", &LtPlan::checked).def_readonly("heuristic_count", &LtPlan::heuristic_count)\n    .def_readonly("diagnostics", &LtPlan::diagnostics)\n    .def_readonly("id_count", &LtPlan::id_count);\n}\n'
tt_lt__LT_EXT = None
tt_lt__LT_WORKSPACE_BYTES = 128 << 20
tt_lt__LT_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
tt_lt__LT_H200_FIXED = {(256, 'o'): (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (128, 'unpooler'): (2, 0, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (256, 'unpooler'): (3, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5]), (128, 'qkv_gate'): (0, 0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (256, 'qkv_gate'): (0, 0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (128, 'pooler'): (2, 0, [66, 15, 1, 0, 0, 4, 35, 0, 2]), (256, 'pooler'): (3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2])}
tt_lt__LT_H200_VERSION = 130101
tt_lt__LT_SEARCH_SEEDS = {(128, 'qkv_gate'): [(2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 4])], (128, 'o'): [(2, 0, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 2, 35, 0, 5]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (128, 'pooler'): [(2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 2, 35, 0, 5])], (128, 'unpooler'): [(2, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 18, 1, 0, 0, 6, 35, 0, 5]), (2, 0, [66, 28, 1, 0, 0, 2, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (256, 'qkv_gate'): [(0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 9]), (1, 0, [66, 134, 1, 0, 0, 2, 35, 0, 9]), (0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 2])], (256, 'o'): [(1, 0, [66, 20, 1, 0, 0, 7, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 4, 35, 0, 5]), (3, 0, [66, 20, 1, 0, 0, 5, 35, 0, 3]), (2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5])], (256, 'pooler'): [(3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 4, 35, 0, 3]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 8]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 9])], (256, 'unpooler'): [(2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 5, 35, 0, 2]), (3, 0, [66, 20, 1, 0, 0, 0, 35, 0, 3])]}

def tt_lt__lt_build():
    global tt_lt__LT_EXT
    if tt_lt__LT_EXT is None:
        tt_lt__LT_EXT = load_inline(name='muse_cublaslt_v1', cpp_sources=tt_lt__LT_CPP, with_cuda=True, extra_cflags=['-O3', '-std=c++20'], extra_ldflags=['-l:libcublasLt.so.13'], verbose=False)
    return tt_lt__LT_EXT

def tt_lt__lt_state(weights: dict) -> dict:
    ext = tt_lt__lt_build()
    device = weights['qkv_gate'].device
    workspace = torch.empty(tt_lt__LT_WORKSPACE_BYTES, device=device, dtype=torch.uint8)
    return {'ext': ext, 'version': ext.version(), 'context': ext.Context(workspace), 'configs': {}, 'weights': {name: weights[name].contiguous() for name in tt_lt__LT_LAYERS}, 'packed': {}, 'workspace': workspace}

def tt_lt__lt_plan(st: dict, layer: str, m: int, layout: int, epilogue: int):
    w = st['weights'][layer]
    n, k = w.shape
    if layout >= 2:
        if layer not in st['packed']:
            st['packed'][layer] = w.t().contiguous()
        w = st['packed'][layer]
    return st['ext'].Plan(st['context'], w, m, n, k, layout, epilogue)

def tt_lt_prepare_cublaslt(weights: dict) -> dict:
    device = weights['qkv_gate'].device
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0) or not tt_lt__LT_H200_FIXED:
        return {'configs': {}}
    st = tt_lt__lt_state(weights)
    if tt_lt__LT_H200_VERSION is not None and st['version'] != tt_lt__LT_H200_VERSION:
        return {'configs': {}}
    for key, descriptor in tt_lt__LT_H200_FIXED.items():
        m, layer = key
        layout, epilogue, config = descriptor
        plan = tt_lt__lt_plan(st, layer, m, layout, epilogue)
        index = plan.restore(list(config))
        if index < 0 or plan.metadata()[index][:9] != list(config):
            continue
        x = torch.zeros((m, st['weights'][layer].shape[1]), device=device, dtype=torch.bfloat16)
        out = torch.empty((m, st['weights'][layer].shape[0]), device=device, dtype=torch.bfloat16)
        if plan.run(x, out, index) == 0:
            st['configs'][key] = (plan, index)
    torch.cuda.synchronize(device)
    return st

def tt_lt_run_cublaslt(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    entry = st['configs'].get((x.shape[0], layer))
    if entry is None or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    w = st['weights'][layer]
    if x.device != w.device or w.dtype != torch.bfloat16 or tuple(w.shape) != (n, x.shape[1]) or x.data_ptr() % 256:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = entry[0].run(x, out, entry[1])
    if rc:
        raise RuntimeError(f'cuBLASLt launch failed: status {rc}')
    return out

"""Experimental lossless BF16 register-A WGMMA projections, one launch per call."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def tt_mw_mma(n):
    nr = n // 2
    regs = ','.join((f'%{i}' for i in range(nr)))
    outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
tt_mw_CUDA_PREFIX = '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t l0,uint32_t l1,uint32_t c,uint32_t t0,uint32_t t1,uint32_t* a) {\n  uint32_t cm=c&0x77777777u,lo=prmt(t0,t1,cm),hi=prmt(t0,t1,cm>>16);\n  a[0]=prmt(l0,lo,0x5140u)|((c<<12)&0x80008000u);\n  a[1]=prmt(l0,lo,0x7362u)|((c<<8)&0x80008000u);\n  a[2]=prmt(l1,hi,0x5140u)|((c<<4)&0x80008000u);\n  a[3]=prmt(l1,hi,0x7362u)|(c&0x80008000u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
tt_mw_CUDA_CORE = '\n\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\n__global__ __launch_bounds__(128*WK,CAP) void m32w_gemm(\n    const __grid_constant__ CUtensorMap tx,const uint4* __restrict__ w,\n    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {\n  extern __shared__ __align__(1024) uint8_t sm[];\n  constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;\n  int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;\n  uint32_t base0=addr(sm),base=base0+wg*WS,full=base+Q*SB;\n  int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;\n  int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;\n  uint64_t pw,px;\n  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));\n  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(px));\n  if(tid%128==0) {\n    for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  auto loadx=[&](int i) {\n    if(tid%128==0 && i<nkb) {\n      int s=i%Q;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n      tma(base+s*SB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<Q;s++)loadx(s);\n  uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;\n  if(C12 && warp<RW) {\n    const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;\n    e0=el[lane];e1=el[lane+32];nxt=__shfl_sync(0xffffffffu,e0,0);\n  }\n  uint4 raw[PF][NL]={};\n  auto load=[&](int slot,int i) {\n    if(i<nkb && warp<RW) {\n      const uint4* src=w+((long)(nt*(k/64)+kb0+i)*RW+warp)*NL*32+lane;\n      #pragma unroll\n      for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);\n    }\n  };\n  #pragma unroll\n  for(int p=0;p<PF;p++)load(p,p);\n  float acc[BN/2];\n  #pragma unroll\n  for(int j=0;j<BN/2;j++)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));\n  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {\n    #pragma unroll\n    for(int p=0;p<PF;p++) {\n      int i=ib+p;if(i>=nkb)break;\n      uint32_t a[16];\n      if constexpr(C12) {\n        dec(raw[p][0].x,raw[p][0].y,raw[p][2].x,t0,t1,a);\n        dec(raw[p][0].z,raw[p][0].w,raw[p][2].y,t0,t1,a+4);\n        dec(raw[p][1].x,raw[p][1].y,raw[p][2].z,t0,t1,a+8);\n        dec(raw[p][1].z,raw[p][1].w,raw[p][2].w,t0,t1,a+12);\n      } else {\n        #pragma unroll\n        for(int l=0;l<4;l++) {\n          a[4*l]=raw[p][l].x;a[4*l+1]=raw[p][l].y;a[4*l+2]=raw[p][l].z;a[4*l+3]=raw[p][l].w;\n        }\n      }\n      load(p,i+PF);\n      if constexpr(C12) {\n        while((nxt>>18)==uint32_t(i)) {\n          if(lane==int((nxt>>13)&31u))patch(a,nxt);\n          ++ep;\n          uint32_t ev=ep<32?e0:e1;\n          nxt=__shfl_sync(0xffffffffu,ev,ep&31);\n          if(ep>=64)nxt=0xffffffffu;\n        }\n      }\n      int s=i%Q;\n      waitbar(full+s*8,(i/Q)&1);\n      #pragma unroll\n      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");\n      if constexpr(BN==16)mma16(acc,a,desc(base+s*SB));\n      if constexpr(BN==32)mma32(acc,a,desc(base+s*SB));\n      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n      #pragma unroll\n      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");\n      loadx(i+Q);\n    }\n  }\n  __syncthreads();\n  if(tid%128==0) {\n    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory");\n  }\n  __syncthreads();\n  int row=warp*16+lane/4,col=2*(lane%4);\n  #pragma unroll\n  for(int j=0;j<BN/8;j++) {\n    #pragma unroll\n    for(int h=0;h<2;h++) {\n      if(warp>=RW)continue;\n      ((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];\n      ((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];\n    }\n  }\n  __syncthreads();\n  if(split>1)cluster_sync();\n  for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split) {\n    int token=q/(R/4),row=(q%(R/4))*4;\n    if(mt*BN+token>=m)continue;\n    float4 sum=make_float4(0,0,0,0);\n    for(int pp=0;pp<split*WK;pp++) {\n      int s=pp/WK;\n      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;\n      if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));\n      float4 v;\n      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(remote):"memory");\n      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;\n    }\n    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);\n    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);\n  }\n  __syncthreads();\n  if(split>1)cluster_sync();\n}\n\n'
tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def tt_mw_cuda_source(instances):
    cases = []
    attrs = []
    for i, inst in enumerate(instances):
        bn, r, wk, pf, q, c, cap = inst
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    return tt_mw_CUDA_PREFIX + '\n'.join((tt_mw_mma(n) for n in (16, 32, 64))) + tt_mw_CUDA_CORE + tt_mw_CUDA_MAP + tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def tt_mw_build(instances):
    src = tt_mw_cuda_source(instances)
    name = 'm32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])
tt_mw_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

def tt_mw__to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_mw__to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_mw_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_mw_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_mw_sm_c12_unpack(data, base, esc):
    """Inverse of sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_mw_SM_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_mw_sm_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = tt_mw_sm_c12_bytes(v)
        if not torch.equal(tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = tt_mw__to_i32(entry)
    else:
        data = tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def tt_mw_launch_raw(ext, idx, pk, x, out):
    return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def tt_mw_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))
tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def tt_mw_prepare_m32w(weights, configs=None) -> dict:
    configs = tt_mw_CONFIGS if configs is None else configs
    device = next(iter(weights.values())).device
    state = {'plans': {}, 'device': device}
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
        return state
    instances = []
    for inst, split in configs.values():
        for c in (inst[5], False):
            twin = inst[:5] + (c,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    ext = tt_mw_build(instances)
    state['ext'] = ext
    packs = {}
    for (m, layer), (inst, split) in configs.items():
        if layer not in weights:
            continue
        w = weights[layer]
        for compressed in (inst[5], False):
            actual = inst[:5] + (compressed,) + inst[6:]
            idx = instances.index(actual)
            if ext.info(idx) % 10000:
                continue
            key = (layer, inst[1], inst[2], split, compressed)
            if key not in packs:
                packs[key] = tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
            pk = packs[key]
            if pk is None or not tt_mw_verify_onehot(ext, idx, pk, w, m):
                continue
            if m == 32 and (not tt_mw_verify_onehot(ext, idx, pk, w, 33)):
                continue
            state['plans'][m, layer] = (idx, pk)
            break
    return state

def tt_mw_run_m32w(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
        return None
    m = x.shape[0]
    if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    key = (16 if m <= 16 else 32, layer)
    plan = st['plans'].get(key)
    if plan is None:
        return None
    idx, pk = plan
    if n != pk['n'] or x.shape[1] != pk['k']:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    rc = tt_mw_launch_raw(st['ext'], idx, pk, x, out)
    if rc:
        raise RuntimeError(f'm32w launch failed: {rc}')
    return out

V9_SELECTED_ROUTES = [{'m': 32, 'layer': 'o', 'name': 'ol_1', 'module': 'ol_v8', 'config': [32, 128, 8, 1, 1, 1]}, {'m': 32, 'layer': 'pooler', 'name': 'cps', 'module': 'cps_v8'}, {'m': 32, 'layer': 'unpooler', 'name': 'mw', 'module': 'mw'}, {'m': 128, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 128, 'layer': 'unpooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'qkv_gate', 'name': 'lt', 'module': 'lt'}]


def prepare_v9(weights: Weights) -> dict:
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights['router'].to(torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = (weights['router']-hi.float()).to(torch.bfloat16).t().contiguous()
    state['router_weight'] = weights['router']
    state['sm'] = prepare_small_m(weights)
    state['m32'] = prepare_m32(weights)
    state['wide2'] = prepare_wide2(weights)
    state['router'] = prepare_router(weights)
    state['qkv'] = prepare_qkv(weights)
    if state['qkv'].get('large') is not None:
        state['qkv']['is_enabled'] = True
    state['router2'] = prepare_router2(weights)
    state['olayer'] = prepare_olayer(weights)
    state['cps'] = prepare_cps(weights)
    state['new_lt'] = tt_lt_prepare_cublaslt(weights)
    state['new_mw'] = tt_mw_prepare_m32w(weights)
    for row in V9_SELECTED_ROUTES:
        m, layer = row['m'], row['layer']
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        x = torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device)
        candidate_v9({'x': x, 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}, state)
    torch.cuda.synchronize(weights['qkv_gate'].device)
    return state





def candidate_sglang(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor:
    if layer != 'router':
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= 8:
        return F.linear(x.float(), state['router_weight'])
    output = torch.mm(x, state['router_hi_t'], out_dtype=torch.float32)
    output += torch.mm(x, state['router_lo_t'], out_dtype=torch.float32)
    return output


def candidate_v8(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    output = None
    if layer == 'router':
        if 2 <= m <= 8:
            output = run_small_m(x, layer, n, state['sm'])
        elif 65 <= m <= 192:
            output = run_router2(x, layer, n, state['router2'])
        if output is None:
            output = run_router(x, layer, n, state['router'])
    elif layer == 'qkv_gate' and m == 128:
        output = run_qkv(x, layer, n, state['qkv'])
    elif layer == 'o' and 65 <= m <= 224:
        output = run_olayer(x, layer, n, state['olayer'])
    elif layer in ('pooler', 'unpooler') and 9 <= m <= 16:
        output = run_cps(x, layer, n, state['cps'])
    if output is None and 9 <= m <= 32:
        output = run_m32(x, layer, n, state['m32'])
    if output is None and m <= 32:
        output = run_small_m(x, layer, n, state['sm'])
    if output is None and 64 < m <= 256:
        output = run_wide2(x, layer, n, state['wide2'])
    return candidate_sglang(x, layer, n, state) if output is None else output





def candidate_base(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    if layer == 'router' and x.shape[0] <= 16:
        output = run_router(x, layer, n, state['router'])
        return candidate_sglang(x, layer, n, state) if output is None else output
    return candidate_v8(data, state)



def candidate_v9(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    if layer == 'o' and m == 32 and n == 8192 and x.shape[1] == 1024 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0:
        output = run_olayer_candidate(x, state['olayer'], OLAYER_CONFIGS.index((32, 128, 8, 1, 1, 1)))
        if output is not None:
            return output
    if layer == 'pooler' and m == 32:
        output = run_cps(x, layer, n, state['cps'])
        if output is not None:
            return output
    if layer == 'unpooler' and m == 32:
        output = tt_mw_run_m32w(x, layer, n, state['new_mw'])
        if output is not None:
            return output
    if layer == 'pooler' and m == 128:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'unpooler' and m == 128:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'pooler' and m == 256:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'qkv_gate' and m == 256:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    return candidate_base(data, state)



"""H200 small-m table refinements. Unclaimed shapes return None."""
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_sm2_SMALLM2_CUDA_TEMPLATE = '\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define SMALLM2D __device__ __forceinline__\n\nSMALLM2D int4 smallm2_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSMALLM2D uint4 smallm2_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSMALLM2D uint32_t smallm2_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nSMALLM2D uint32_t smallm2_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nSMALLM2D void smallm2_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nSMALLM2D void smallm2_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = smallm2_prmt(T0, T1, cm);\n  const uint32_t hhi = smallm2_prmt(T0, T1, cm >> 16);\n  a[0] = smallm2_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = smallm2_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = smallm2_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = smallm2_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\n__device__ __noinline__ void smallm2_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nSMALLM2D void smallm2_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) smallm2_never(p);\n}\n\nSMALLM2D uint32_t smallm2_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nSMALLM2D uint32_t smallm2_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nSMALLM2D uint32_t smallm2_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nSMALLM2D void smallm2_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nSMALLM2D void smallm2_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nSMALLM2D void smallm2_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nSMALLM2D void smallm2_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nSMALLM2D void smallm2_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nSMALLM2D void smallm2_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nSMALLM2D bool smallm2_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nSMALLM2D uint64_t smallm2_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a wrong result instead of a hang.\nSMALLM2D void smallm2_wait_bar(uint32_t bar) {\n  if (smallm2_mbar_try(bar)) return;\n  const uint64_t t0 = smallm2_gtime();\n  while (!smallm2_mbar_try(bar))\n    if (smallm2_gtime() - t0 > 20000000ull) break;\n}\n\nSMALLM2D void smallm2_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nSMALLM2D float4 smallm2_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nSMALLM2D void smallm2_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nSMALLM2D void smallm2_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct Smallm2Fmt;\ntemplate <> struct Smallm2Fmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct Smallm2Fmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct Smallm2Fmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct Smallm2Fmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nSMALLM2D void smallm2_decode(const int4 (&w)[Smallm2Fmt<FMT>::NL], uint32_t (&A)[Smallm2Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < Smallm2Fmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < Smallm2Fmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      smallm2_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      smallm2_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      smallm2_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      smallm2_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nSMALLM2D void smallm2_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nSMALLM2D void smallm2_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) smallm2_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n__global__ void __launch_bounds__(NTMAX) smallm2_gemv(\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = Smallm2Fmt<FMT>::NL, NA = Smallm2Fmt<FMT>::NA;\n  constexpr bool C12 = FMT == 1 || FMT == 3;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);\n    const uint32_t bar = smallm2_smem_u32(&s_bar);\n    uint32_t rank = 0, my_e = 0xFFFFFFFFu;\n    if constexpr (!(OPT & 1)) {\n    if constexpr (PUSH) {\n      rank = smallm2_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        smallm2_mbar_init(bar, 1);\n        smallm2_mbar_expect(bar, parts * RT * TV * 128u);\n        smallm2_fence_init();\n      }\n      smallm2_cl_arrive();\n    }\n\n    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);\n\n    }\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) w[slot][j][l] = smallm2_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO && !(OPT & 2)) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = smallm2_ldx(xp);\n          xv[slot][n][1] = smallm2_ldx(xp + 8);\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n    if constexpr (OPT & 1) {\n    if constexpr (PUSH) {\n      rank = smallm2_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        smallm2_mbar_init(bar, 1);\n        smallm2_mbar_expect(bar, parts * RT * TV * 128u);\n        smallm2_fence_init();\n      }\n      smallm2_cl_arrive();\n    }\n\n    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);\n\n    }\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n        } else {\n          uint32_t b[MT][8];\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else if constexpr (OPT & 2) {\n              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;\n              uint4 u = smallm2_ldx(xp), v = smallm2_ldx(xp + 8);\n              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;\n              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;\n            } else {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) smallm2_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n          if (si + ST < KU) load(s, si + ST);\n          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) smallm2_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n                } else {\n                  smallm2_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                  if constexpr (HILO) smallm2_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                }\n              }\n        }\n      }\n    }\n\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          smallm2_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) smallm2_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                smallm2_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          if constexpr (OPT & 8) { emit_full(idx, v); return; }\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem);\n          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)\n              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });\n          } else {\n            float4* csum = part + (size_t)NW * RT * MT * 32;\n            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) smallm2_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2) {\n            smallm2_cl_wait();\n            if (t < TV) {\n              const uint32_t rbar = smallm2_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = n * 4 + t;\n                  if (tn < TV) {\n                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            smallm2_wait_bar(bar);\n            smallm2_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if (n * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            smallm2_cl_wait();\n            const uint32_t rbar = smallm2_mapa(bar, 0);\n            smallm2_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            smallm2_wait_bar(bar);\n            smallm2_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int smallm2_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\nstatic int smallm2_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = smallm2_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * 8) return -5;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = smallm2_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[1];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    lc.numAttrs = 1;\n  }\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\n@@CASES@@\n    default: return -99;\n  }\n}\n\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\ntemplate<int M, int NT, int ST>\n__global__ void smallm2_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,\n                                float* __restrict__ y, int m) {\n  constexpr int K=8192;\n  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n  float acc[M][4]={};\n  float4 ww[ST];\n  uint2 xx[ST][M];\n  auto ld=[&](int slot,int i) {\n    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n    #pragma unroll\n    for(int t=0;t<M;t++) {\n      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n  #pragma unroll 1\n  for(int base=tid;base<K/4;base+=ST*NT) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      float4 v=ww[s];\n      uint2 a[M];\n      #pragma unroll\n      for(int t=0;t<M;t++) a[t]=xx[s][t];\n      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n      #pragma unroll\n      for(int t=0;t<M;t++) if(t<m) {\n        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);\n        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);\n        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);\n        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);\n        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);\n        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);\n        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);\n      }\n    }\n  }\n  __shared__ float sums[M][NT/32];\n  #pragma unroll\n  for(int t=0;t<M;t++) if(t<m) {\n    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n    #pragma unroll\n    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n    if(lane==0) sums[t][warp]=v;\n  }\n  __syncthreads();\n  if(warp==0) {\n    #pragma unroll\n    for(int t=0;t<M;t++) if(t<m) {\n      float v=lane<NT/32?sums[t][lane]:0.f;\n      #pragma unroll\n      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n      if(lane==0) y[t*256+row]=v;\n    }\n  }\n}\n\nint smallm2_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n  auto kern=smallm2_router_scalar<1,512,2>;\n  if(query==2) {\n    cudaFuncAttributes fa;\n    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;\n    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);\n  }\n  if(m!=1) return -5;\n  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);\n  return cudaGetLastError()==cudaSuccess?0:-8;\n}\n'
tt_v10_sm2_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v10_sm2_SMALLM2_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
tt_v10_sm2_SMALLM2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v10_sm2_SMALLM2_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
tt_v10_sm2_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_sm2_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v10_sm2_SMALLM2_ESC_CAP = 32

def tt_v10_sm2_smallm2_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def tt_v10_sm2_smallm2_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
        cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return tt_v10_sm2_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v10_sm2_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, tt_v10_sm2_SMALLM2_CPP + extra_cpp, cuda_sources=tt_v10_sm2_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v10_sm2_smallm2_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_sm2_smallm2_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_sm2_smallm2_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_sm2_smallm2_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v10_sm2_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_sm2_smallm2_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_v10_sm2_smallm2_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_v10_sm2_smallm2_c12_unpack(data, base, esc):
    """Inverse of smallm2_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_v10_sm2_smallm2_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_sm2_smallm2_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v10_sm2_smallm2_units(hi), tt_v10_sm2_smallm2_units(lo)]
    else:
        mats = [tt_v10_sm2_smallm2_units(w)]
    if fmt in (0, 2):
        data = torch.cat([tt_v10_sm2_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v10_sm2_smallm2_c12_bytes(v)
        if not torch.equal(tt_v10_sm2_smallm2_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(tt_v10_sm2_smallm2_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_sm2_smallm2_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v10_sm2_SMALLM2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v10_sm2_SMALLM2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v10_sm2_smallm2_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp['fmt'] == 4:
        return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_sm2_smallm2_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v10_sm2_smallm2_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v10_sm2_smallm2_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v10_sm2_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_sm2_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v10_sm2_smallm2_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def tt_v10_sm2_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = tt_v10_sm2_smallm2_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = tt_v10_sm2_smallm2_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = tt_v10_sm2_smallm2_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = tt_v10_sm2_smallm2_verify_onehot(ext, inst_id[inst], sp, tt_v10_sm2_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v10_sm2_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
tt_v10_sm2_SMALLM2_ROUTES = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1)), ('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

def tt_v10_sm2_prepare_smallm2(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in tt_v10_sm2_SMALLM2_ROUTES.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
    ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_final_v3')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

def tt_v10_sm2_run_smallm2(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != tt_v10_sm2_SMALLM2_LAYER_K[layer]:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = tt_v10_sm2_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None
tt_v10_sm2_SMALLM2_ROUTES = {key: value for key, value in tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}

"""Hopper BF16 projections tuned with cold-L2 H200 graph replays."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_tune_CUDA = '\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) {return (uint32_t)__cvta_generic_to_shared(p);}\nD void waitbar(uint32_t b,int ph) {\n  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(b),"r"(ph):"memory");\n}\nD void arrive(uint32_t b) {asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(b):"memory");}\nD void cluster_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}\nD uint32_t remote(uint32_t p,int rank) {uint32_t q;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(q):"r"(p),"r"(rank));return q;}\nD void wait_remote(uint32_t b) {\n  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],0; @!p bra L;}"::"r"(b):"memory");\n}\nD void push(uint32_t p,float4 v,uint32_t b) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");\n}\nD uint64_t desc(uint32_t a) {return uint64_t((a&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t p) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(p):"memory");\n}\n@@MMA@@\ntemplate<int BN,int WG,int ST,int PACK,int SPLIT,int PUSH,int PW,int PX,int HINT,int ORDER,int SW,int PROD>\n__global__ __launch_bounds__((WG+PROD)*128,1) void projection(\n    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,\n    const uint8_t* w,__nv_bfloat16* out,int m,int n,int k) {\n  constexpr int R=64*WG,CT=128*WG,TH=128*(WG+PROD),AB=R*128,SB=(R+BN)*128;\n  constexpr int JP=(BN/8+SPLIT-1)/SPLIT;\n  extern __shared__ __align__(1024) uint8_t sm[];\n  uint32_t base=addr(sm),full=base+ST*SB,empty=full+ST*8,rb=empty+ST*8,drained=rb+8;\n  int tid=threadIdx.x,rank=blockIdx.x%SPLIT,tile=blockIdx.x/SPLIT;\n  int mts=(m+BN-1)/BN,nts=n/R,mt,nt;\n  if constexpr(ORDER==0) {mt=tile%mts;nt=tile/mts;}\n  else {nt=tile%nts;mt=tile/nts;}\n  if constexpr(SW>0) nt=(nt*SW)%nts;\n  int kb0=(k/64)*rank/SPLIT,nkb=(k/64)*(rank+1)/SPLIT-kb0;\n  int own=min(JP,BN/8-rank*JP);\n  uint64_t pw,px;\n  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pw));px=pw;\n  if constexpr(HINT&1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));\n  if constexpr(HINT&2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(px));\n  auto load=[&](int i) {\n    int s=i%ST;\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(full+s*8),"r"(SB):"memory");\n    if constexpr(PACK) {\n      const uint8_t* src=w+((long)nt*(k/64)+kb0+i)*AB;\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;"::"r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw):"memory");\n    } else tma(base+s*SB,&tw,(kb0+i)*64,nt*R,full+s*8,pw);\n    tma(base+s*SB+AB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);\n  };\n  if(tid==0) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      asm volatile("mbarrier.init.shared::cta.b64 [%0],1; mbarrier.init.shared::cta.b64 [%1],%2;"::"r"(full+s*8),"r"(empty+s*8),"n"(WG):"memory");\n    }\n    if constexpr(PUSH && SPLIT>1) {\n      asm volatile("mbarrier.init.shared::cta.b64 [%0],1; mbarrier.init.shared::cta.b64 [%1],%2;"::"r"(rb),"r"(drained),"n"(SPLIT-1):"memory");\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(rb),"r"((SPLIT-1)*own*CT*16):"memory");\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\n  }\n  __syncthreads();\n  if constexpr(PUSH && SPLIT>1) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");\n  if constexpr(!PROD) if(tid==0) for(int i=0;i<ST && i<nkb;i++) load(i);\n  if(PROD && tid>=CT) {\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n    if(tid==CT) for(int i=0;i<nkb;i++) {\n      if(i>=ST) waitbar(empty+(i%ST)*8,((i/ST)-1)&1);\n      load(i);\n    }\n    if constexpr(PUSH && SPLIT>1) {asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");return;}\n  } else {\n    if constexpr(PROD) asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");\n    float a[BN/2];\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;":"=f"(a[j])::"memory");\n    #pragma unroll 1\n    for(int i=0;i<nkb;i++) {\n      int s=i%ST;\n      waitbar(full+s*8,(i/ST)&1);\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("":"+f"(a[j])::"memory");\n      asm volatile("wgmma.fence.sync.aligned;":::"memory");\n      #pragma unroll\n      for(int kk=0;kk<4;kk++) {\n        uint64_t aw=desc(base+s*SB+(tid/128)*8192+kk*32),ax=desc(base+s*SB+AB+kk*32);\n        @@CALL@@\n      }\n      asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("":"+f"(a[j])::"memory");\n      asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\n      if(i>0 && tid%128==0) {\n        if constexpr(PROD) arrive(empty+((i-1)%ST)*8);\n        else if(i-1+ST<nkb) load(i-1+ST);\n      }\n    }\n    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(a[j])::"memory");\n    asm volatile("bar.sync 1,%0;"::"n"(CT):"memory");\n    if constexpr(PUSH && SPLIT>1) {\n      asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");\n      if(tid==0) for(int r=0;r<SPLIT;r++) if(r!=rank)\n        asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote(drained,r)):"memory");\n      wait_remote(drained);\n      #pragma unroll\n      for(int j=0;j<BN/8;j++) if(j/JP!=rank)\n        push(remote(base+((rank*JP+j%JP)*CT+tid)*16,j/JP),make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]),remote(rb,j/JP));\n      wait_remote(rb);\n      #pragma unroll\n      for(int j=0;j<BN/8;j++) if(j/JP==rank) {\n        float4 sum=make_float4(0,0,0,0);\n        #pragma unroll\n        for(int r=0;r<SPLIT;r++) {\n          float4 v=r==rank?make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]):((float4*)sm)[(r*JP+j%JP)*CT+tid];\n          sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;\n        }\n        a[4*j]=sum.x;a[4*j+1]=sum.y;a[4*j+2]=sum.z;a[4*j+3]=sum.w;\n      }\n      asm volatile("bar.sync 1,%0;"::"n"(CT):"memory");\n    }\n    #pragma unroll\n    for(int j=0;j<BN/8;j++) {\n      if constexpr(SPLIT==1 || PUSH) {\n        if(!PUSH || SPLIT==1 || j/JP==rank) {\n          __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]),hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n          int row=(tid/128)*64+((tid%128)/32)*16+((tid%16)/8)*8,col=8*j+tid%8;\n          asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"::"r"(base+2*(col*(R+8)+row)),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi):"memory");\n        }\n      } else {\n        int row=(tid/128)*64+((tid%128)/32)*16+(tid%32)/4,col=2*(tid%4)+8*j;\n        ((float*)sm)[col*(R+4)+row]=a[4*j];((float*)sm)[(col+1)*(R+4)+row]=a[4*j+1];\n        ((float*)sm)[col*(R+4)+row+8]=a[4*j+2];((float*)sm)[(col+1)*(R+4)+row+8]=a[4*j+3];\n      }\n    }\n    if constexpr(PUSH && SPLIT>1) {\n      asm volatile("bar.sync 1,%0;"::"n"(CT):"memory");\n      for(int q=tid;q<own*8*(R/8);q+=CT) {\n        int token=rank*JP*8+q/(R/8),row=q%(R/8)*8;\n        if(mt*BN+token<m) *(uint4*)(out+(long)(mt*BN+token)*n+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*(R+8)+row);\n      }\n      return;\n    }\n  }\n  __syncthreads();\n  if constexpr(SPLIT==1) {\n    for(int q=tid;q<BN*(R/8);q+=TH) {\n      int token=q/(R/8),row=q%(R/8)*8;\n      if(mt*BN+token<m) *(uint4*)(out+(long)(mt*BN+token)*n+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*(R+8)+row);\n    }\n  } else {\n    cluster_sync();\n    for(int q=tid+rank*TH;q<BN*(R/4);q+=TH*SPLIT) {\n      int token=q/(R/4),row=q%(R/4)*4;\n      if(mt*BN+token>=m) continue;\n      float4 sum=make_float4(0,0,0,0);\n      #pragma unroll\n      for(int r=0;r<SPLIT;r++) {\n        float4 v;uint32_t p=remote(base+(token*(R+4)+row)*4,r);\n        asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(p):"memory");\n        sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;\n      }\n      __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);\n      *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);\n    }\n    __syncthreads();cluster_sync();\n  }\n}\nusing Encode=decltype(&cuTensorMapEncodeTiled);\nint encode_map(CUtensorMap* tm,const void* p,int rows,int k,int box,int promo) {\n  static Encode fn=nullptr;\n  if(!fn) {cudaDriverEntryPointQueryResult q;auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&fn,12000,cudaEnableDefault,&q);if(rc!=cudaSuccess||!fn)return -100;}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},strides[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(fn(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,strides,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,(CUtensorMapL2promotion)promo,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\ntemplate<int BN,int WG,int ST,int PACK,int SPLIT,int PUSH,int PW,int PX,int HINT,int ORDER,int SW,int PROD>\nint launch_t(uint64_t x,uint64_t w,uint64_t y,int m,int n,int k,uint64_t stream,int query) {\n  constexpr int R=64*WG,MAIN=ST*((R+BN)*128+16)+16,JP=(BN/8+SPLIT-1)/SPLIT;\n  constexpr int EPI=PUSH?SPLIT*JP*(WG*128)*16:BN*(R+4)*4,SMEM=MAIN>EPI?MAIN:EPI;\n  auto fn=projection<BN,WG,ST,PACK,SPLIT,PUSH,PW,PX,HINT,ORDER,SW,PROD>;\n  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}\n  if(m<1||m>256||n%R||k%64||(SW && (n/R)%5==0))return -1;\n  CUtensorMap tx,tw;int rc=encode_map(&tx,(void*)x,m,k,BN,PX);if(rc)return rc;rc=encode_map(&tw,(void*)w,n,k,R,PW);if(rc)return rc;\n  static bool ready=false;\n  if(!ready) {\n    rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;\n    rc=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(rc)return rc;ready=true;\n  }\n  cudaLaunchConfig_t c={};c.gridDim=dim3((n/R)*((m+BN-1)/BN)*SPLIT);c.blockDim=dim3((WG+PROD)*128);c.dynamicSmemBytes=SMEM;c.stream=(cudaStream_t)stream;\n  cudaLaunchAttribute a={};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim={SPLIT,1,1};if(SPLIT>1){c.attrs=&a;c.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&c,fn,tw,tx,(const uint8_t*)w,(__nv_bfloat16*)y,m,n,k));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t y,int m,int n,int k,uint64_t stream,int query) {\n  switch(id) {@@CASES@@ default:return -99;}\n}\n'
tt_v10_tune_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,int,int,int,uint64_t,int);\n'

def tt_v10_tune_source(configs: list) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'D void mma{n}(float* a,uint64_t w,uint64_t x) {{asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;":{outs}:"l"(w),"l"(x));}}')
    cases = '\n'.join((f"case {i}:return launch_t<{','.join(map(str, c))}>(x,w,y,m,n,k,stream,query);" for i, c in enumerate(configs)))
    return tt_v10_tune_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CALL@@', '\n'.join((f'if constexpr(BN=={n}) mma{n}(a,aw,ax);' for n in ns))).replace('@@CASES@@', cases)

def tt_v10_tune_build(configs: list):
    for c in configs:
        bn, wg, st, pack, split, push, pw, px, hint, order, sw, prod = c
        assert bn in (128, 192, 256) and wg in (1, 2) and (st >= 2) and (split in (1, 2, 3, 4))
        assert st * ((64 * wg + bn) * 128 + 16) + 16 <= 227328
        assert prod in (0, 1) and (prod or wg == 1)
        assert pack in (0, 1) and push in (0, 1) and (pw in range(4)) and (px in range(4))
        assert hint in range(4) and order in (0, 1) and (sw in (0, 5))
        if push and split > 1:
            assert split * ((bn // 8 + split - 1) // split) * (wg * 128) * 16 <= st * (64 * wg + bn) * 128
    src = tt_v10_tune_source(configs)
    name = 'm256tune_' + hashlib.sha256(src.encode()).hexdigest()[:16]
    return load_inline(name, tt_v10_tune_CPP, cuda_sources=src, functions=['launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=True, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def tt_v10_tune_pack_weight(w: torch.Tensor, rows: int) -> torch.Tensor:
    n, k = w.shape
    col = torch.arange(64, device=w.device)[None, :] ^ (torch.arange(rows, device=w.device)[:, None] & 7) * 8
    return w.reshape(n // rows, rows, k // 64, 64).permute(0, 2, 1, 3).gather(-1, col.expand(n // rows, k // 64, rows, 64)).reshape(n, k).contiguous()

def tt_v10_tune_run_candidate(x: torch.Tensor, w: torch.Tensor, ext, index: int, n: int) -> torch.Tensor:
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = ext.launch(index, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], n, x.shape[1], torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'm256tune launch failed {rc}')
    return out
tt_v10_tune_CONFIGS = []
tt_v10_tune_ROUTES = {}

def tt_v10_tune_prepare_m256tune(weights: dict) -> dict:
    state = {'routes': {}}
    if not tt_v10_tune_CONFIGS:
        return state
    device = next(iter(weights.values())).device
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
        return state
    state.update(ext=tt_v10_tune_build(tt_v10_tune_CONFIGS), weights={}, routes=tt_v10_tune_ROUTES)
    for layer, index in tt_v10_tune_ROUTES.items():
        w = weights[layer].contiguous()
        cfg = tt_v10_tune_CONFIGS[index]
        state['weights'][layer] = tt_v10_tune_pack_weight(w, 64 * cfg[1]) if cfg[3] else w
    return state

def tt_v10_tune_run_m256tune(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    index = st['routes'].get(layer)
    if index is None or x.ndim != 2 or (not 129 <= x.shape[0] <= 256):
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda or (not x.is_contiguous()) or x.data_ptr() % 16:
        return None
    w = st['weights'][layer]
    if x.device != w.device or tuple(w.shape) != (n, x.shape[1]) or w.dtype != torch.bfloat16:
        return None
    return tt_v10_tune_run_candidate(x, w, st['ext'], index, n)
tt_v10_tune_RL_CUDA = '\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define OD __device__ __forceinline__\nOD uint32_t rl_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\nOD void rl_wait(uint32_t b,int ph) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");\n}\nOD void rl_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nOD uint64_t rl_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nOD void rl_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {\n  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");\n}\n@@MMA@@\n\ntemplate<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void rl_early_projection(\n    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,\n    const uint8_t* w,__nv_bfloat16* out,int m) {\n  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;\n  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);\n  extern __shared__ __align__(1024) uint8_t sm[];\n  uint32_t base=rl_addr(sm),full=base+ST*SB,empty=full+ST*8;\n  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;\n  uint64_t pw,px;\n  if constexpr(HINT) {\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));\n    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));\n  } else {\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;\n  }\n  auto load=[&](int i) {\n    int s=i%ST;\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n    if constexpr(PACK==1) {\n      const uint8_t* src=w+((long)nt*IT+i)*AB;\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");\n    } else if constexpr(PACK==2) rl_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);\n    else rl_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);\n    rl_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);\n  };\n  if(tid==((PIPE&2)?CT:0)) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  if constexpr(PIPE&4) __syncthreads();\n  if(tid==((PIPE&2)?CT:0)) {\n    #pragma unroll\n    for(int i=0;i<ST && i<IT;i++) load(i);\n  }\n  if constexpr(!(PIPE&4)) __syncthreads();\n  if((PIPE&2) && tid>=CT) {\n    if(tid==CT) {\n      #pragma unroll 1\n      for(int i=ST;i<IT;i++) {rl_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}\n    }\n  } else {\n    float a[BN/2];\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");\n    #pragma unroll 1\n    for(int i=0;i<IT;i++) {\n      int s=i%ST;\n      rl_wait(full+s*8,(i/ST)&1);\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\n      #pragma unroll\n      for(int kk=0;kk<BK/16;kk++) {\n        uint64_t aw=rl_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);\n        uint64_t ax=rl_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);\n        @@CALL@@\n      }\n      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");\n      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");\n      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n      int done=i-((PIPE&1)?1:0);\n      if(done>=0 && done+ST<IT && tid%128==0) {\n        if constexpr(PIPE&2) rl_arrive(empty+(done%ST)*8);\n        else load(done+ST);\n      }\n    }\n    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");\n    #pragma unroll\n    for(int j=0;j<BN/8;j++) {\n      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);\n      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);\n      uint32_t d=base+2*(cb*72+rb);\n      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");\n    }\n  }\n  __syncthreads();\n  for(int q=tid;q<T*(R/8);q+=TH) {\n    int token=q/(R/8),row=(q%(R/8))*8;\n    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);\n  }\n}\n\nusing OLEncode=decltype(&cuTensorMapEncodeTiled);\nint rl_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {\n  static OLEncode encode=nullptr;\n  if(!encode) {\n    cudaDriverEntryPointQueryResult q;\n    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);\n    if(rc!=cudaSuccess || !encode) return -100;\n  }\n  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};\n  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\ntemplate<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\nint rl_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n  constexpr int T=BN*TW,R=PACK==2?32:64;\n  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;\n  auto fn=rl_early_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;\n  static bool configured=false;\n  if(!configured) {\n    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;\n    configured=true;\n  }\n  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}\n  CUtensorMap tw,tx;\n  int rc=rl_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;\n  rc=rl_map(&tx,(void*)x,m,T,BK);if(rc)return rc;\n  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);\n  return int(cudaGetLastError());\n}\nint rl_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n  switch(id) {@@CASES@@ default:return -99;}\n}\n'
tt_v10_tune_RL_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rl_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def tt_v10_tune_rl_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void rl_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return rl_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) rl_mma{n}(a,aw,ax);' for n in ns))
    return tt_v10_tune_RL_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def tt_v10_tune_build_rl(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (2, 3, 6, 7, 14, 15)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported rl configuration: {cfg}')
    tt_v10_tune_source = tt_v10_tune_rl_source(configs)
    name = 'rl_' + hashlib.sha256(tt_v10_tune_source.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_tune_RL_CPP, cuda_sources=tt_v10_tune_source, functions=['rl_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

"""Lossless BF16 projection refinements; only confirmed exact shapes are claimed."""
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_warps_W0_DC2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v10_warps_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_warps_W0_DC2_ESC_CAP = 32

def tt_v10_warps_wp0_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_warps_wp0_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_warps_wp0_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_warps_wp0_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v10_warps_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_warps_wp0_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_v10_warps_wp0_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_v10_warps_wp0_c12_unpack(data, base, esc, decode=0):
    """Inverse of wp0_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_v10_warps_wp0_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_warps_wp0_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v10_warps_wp0_units(hi), tt_v10_warps_wp0_units(lo)]
    else:
        mats = [tt_v10_warps_wp0_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([tt_v10_warps_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v10_warps_wp0_c12_bytes(v, decode)
        if not torch.equal(tt_v10_warps_wp0_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(tt_v10_warps_wp0_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_warps_wp0_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v10_warps_W0_DC2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v10_warps_W0_DC2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v10_warps_wp0_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
tt_v10_warps_W0_CUDA = '\n#define DC2_DECODE 5\n#define DC2_MIN_CTAS 1\n\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define DC2D __device__ __forceinline__\n\nDC2D int4 wp_bde7122f9b4f_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nDC2D uint4 wp_bde7122f9b4f_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nDC2D uint32_t wp_bde7122f9b4f_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nDC2D uint32_t wp_bde7122f9b4f_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nDC2D void wp_bde7122f9b4f_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nDC2D void wp_bde7122f9b4f_dec_old(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = wp_bde7122f9b4f_prmt(T0, T1, cm);\n  const uint32_t hhi = wp_bde7122f9b4f_prmt(T0, T1, cm >> 16);\n  a[0] = wp_bde7122f9b4f_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = wp_bde7122f9b4f_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = wp_bde7122f9b4f_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = wp_bde7122f9b4f_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\n\nDC2D uint32_t wp_bde7122f9b4f_mul(uint32_t a, uint32_t scale) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(scale));\n  return d;\n}\n\nDC2D uint32_t wp_bde7122f9b4f_mask_bias(uint32_t a) {\n  uint32_t d;\n  asm("lop3.b32 %0, %1, 0x87ff87ff, 0x40004000, 0xea;" : "=r"(d) : "r"(a));\n  return d;\n}\n\nDC2D void wp_bde7122f9b4f_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n#if DC2_DECODE == 0\n  wp_bde7122f9b4f_dec_old(L0, L1, C, T0, T1, a);\n#elif DC2_DECODE == 1\n  wp_bde7122f9b4f_dec_old(L0, L1, C, 0x43424140u, 0x47464544u, a);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) a[q] = wp_bde7122f9b4f_mul(a[q], T0);\n#else\n#if DC2_DECODE >= 4\n  uint32_t hi;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(hi) : "r"(C));\n#else\n  const uint32_t hi = ((C >> 4) & 0x07070707u) | ((C << 4) & 0x80808080u);\n#endif\n  a[0] = wp_bde7122f9b4f_prmt(L0, C, 0x5140u);\n  a[1] = wp_bde7122f9b4f_prmt(L0, C, 0x7362u);\n  a[2] = wp_bde7122f9b4f_prmt(L1, hi, 0x5140u);\n  a[3] = wp_bde7122f9b4f_prmt(L1, hi, 0x7362u);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) {\n#if DC2_DECODE == 2 || DC2_DECODE == 4\n    a[q] = wp_bde7122f9b4f_mul(wp_bde7122f9b4f_mask_bias(a[q]), T0);\n#else\n    a[q] = (a[q] & 0x87ff87ffu) + T0;\n#endif\n  }\n#endif\n}\n\n__device__ __noinline__ void wp_bde7122f9b4f_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nDC2D void wp_bde7122f9b4f_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) wp_bde7122f9b4f_never(p);\n}\n\nDC2D uint32_t wp_bde7122f9b4f_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nDC2D uint32_t wp_bde7122f9b4f_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nDC2D uint32_t wp_bde7122f9b4f_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nDC2D void wp_bde7122f9b4f_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nDC2D void wp_bde7122f9b4f_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nDC2D void wp_bde7122f9b4f_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nDC2D void wp_bde7122f9b4f_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nDC2D void wp_bde7122f9b4f_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nDC2D void wp_bde7122f9b4f_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nDC2D bool wp_bde7122f9b4f_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nDC2D uint64_t wp_bde7122f9b4f_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a wrong result instead of a hang.\nDC2D void wp_bde7122f9b4f_wait_bar(uint32_t bar) {\n  if (wp_bde7122f9b4f_mbar_try(bar)) return;\n  const uint64_t t0 = wp_bde7122f9b4f_gtime();\n  while (!wp_bde7122f9b4f_mbar_try(bar))\n    if (wp_bde7122f9b4f_gtime() - t0 > 20000000ull) break;\n}\n\nDC2D void wp_bde7122f9b4f_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nDC2D float4 wp_bde7122f9b4f_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nDC2D void wp_bde7122f9b4f_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nDC2D void wp_bde7122f9b4f_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct DC2Fmt;\ntemplate <> struct DC2Fmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct DC2Fmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct DC2Fmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct DC2Fmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nDC2D void wp_bde7122f9b4f_decode(const int4 (&w)[DC2Fmt<FMT>::NL], uint32_t (&A)[DC2Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < DC2Fmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < DC2Fmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      wp_bde7122f9b4f_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      wp_bde7122f9b4f_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      wp_bde7122f9b4f_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      wp_bde7122f9b4f_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nDC2D void wp_bde7122f9b4f_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nDC2D void wp_bde7122f9b4f_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) wp_bde7122f9b4f_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>\n#if DC2_MIN_CTAS > 1\n__global__ void __launch_bounds__(NTMAX, DC2_MIN_CTAS) wp_bde7122f9b4f_gemv(\n#else\n__global__ void __launch_bounds__(NTMAX) wp_bde7122f9b4f_gemv(\n#endif\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = DC2Fmt<FMT>::NL, NA = DC2Fmt<FMT>::NA, MTT = MT * TG;\n  constexpr bool C12 = FMT == 1 || FMT == 3;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;\n  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);\n  const int wbytes = XM == 4 ? CWR * CWK * RT * NA * 2048 : 0;\n  int4* sw = reinterpret_cast<int4*>(smem + xbytes);\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;\n    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = min(4 * MTT, (m + 1) >> 1);\n    const uint32_t bar = wp_bde7122f9b4f_smem_u32(&s_bar);\n    uint32_t rank = 0;\n    if constexpr (PUSH) {\n      rank = wp_bde7122f9b4f_rank();\n      if (rank == 0) {\n        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        if (tid == 0) { wp_bde7122f9b4f_mbar_init(bar, 1); wp_bde7122f9b4f_fence_init(); }\n        __syncthreads();\n        if (tid == 0) { wp_bde7122f9b4f_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u); wp_bde7122f9b4f_fence_init(); }\n      }\n      wp_bde7122f9b4f_cl_arrive();\n    }\n\n    if constexpr (XM == 2 || XM == 3 || XM == 5) {\n      const int ks = K / S;\n      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {\n        const int t = v / (ks / 8), c = v % (ks / 8);\n        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;\n        void* dst = sx + t * (ks + 8) + c * 8;\n        if constexpr (XM == 3 || XM == 5) {\n          const uint32_t sa = wp_bde7122f9b4f_smem_u32(dst);\n          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");\n        } else *reinterpret_cast<uint4*>(dst) = wp_bde7122f9b4f_ldx(src);\n      }\n      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");\n      else __syncthreads();\n    }\n    uint32_t my_e = 0xFFFFFFFFu;\n    if constexpr (C12 && !RO) my_e = wp_bde7122f9b4f_lde(esc + (size_t)gw * 32 + lane);\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);\n\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) if (XM != 4 || tg == 0) w[slot][j][l] = wp_bde7122f9b4f_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = wp_bde7122f9b4f_ldx(xp);\n          xv[slot][n][1] = wp_bde7122f9b4f_ldx(xp + (XM >= 9 ? 32 : 8));\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    wp_bde7122f9b4f_order_point(m, outp);\n    if constexpr (XM == 3 || XM == 5) {\n      asm volatile("cp.async.wait_group 0;" ::: "memory");\n      __syncthreads();\n    }\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          wp_bde7122f9b4f_order_point(m, outp);\n        } else {\n          uint32_t b[MT][8];\n          if constexpr (XM == 7 || XM == 8 || XM == 11) {\n#pragma unroll\n            for (int n = 0; n < MT; ++n) {\n              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;\n              const uint4 z0=wp_bde7122f9b4f_ldx(xp), z1=wp_bde7122f9b4f_ldx(xp+(XM >= 9 ? 32 : 8));\n              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;\n              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;\n            }\n            wp_bde7122f9b4f_order_point(m, outp);\n          }\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) wp_bde7122f9b4f_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          if (si + ST < KU) load(s, si + ST);\n          wp_bde7122f9b4f_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) wp_bde7122f9b4f_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n          if constexpr (XM == 4) {\n            if (tg == 0) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int q = 0; q < NA * 4; ++q)\n                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =\n                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);\n            }\n            __syncthreads();\n            if (tg != 0) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int q = 0; q < NA * 4; ++q) {\n                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];\n                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;\n                }\n            }\n          }\n          if constexpr (XM == 6 || XM == 8 || XM == 10) {\n#pragma unroll\n            for (int q = 0; q < 4; ++q)\n#pragma unroll\n              for (int n = 0; n < MT; ++n)\n#pragma unroll\n                for (int j = 0; j < RT; ++j)\n                  wp_bde7122f9b4f_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);\n          } else if constexpr (XM == 5) {\n#pragma unroll\n            for (int q = 0; q < 4; ++q) {\n#pragma unroll\n              for (int np = 0; np < (MT + 1) / 2; ++np) {\n                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);\n                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;\n                const uint32_t addr = wp_bde7122f9b4f_smem_u32(sx + tok * (K / S + 8) + col);\n                uint32_t b0,b1,b2,b3;\n                if constexpr (MT == 1)\n                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));\n                else\n                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));\n#pragma unroll\n                for (int j = 0; j < RT; ++j) {\n                  wp_bde7122f9b4f_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);\n                  if constexpr (MT > 1) wp_bde7122f9b4f_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);\n                }\n              }\n            }\n          } else {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n) {\n              uint32_t bx[8];\n              if constexpr (XM >= 1 && XM <= 4 && !NOX) {\n                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);\n                const __nv_bfloat16* xp;\n                if constexpr (XM == 2 || XM == 3 || XM == 5)\n                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;\n                else xp = x0 + (size_t)tok * K + si * 64;\n                uint4 z0, z1;\n                if constexpr (XM == 2 || XM == 3 || XM == 5) {\n                  z0 = *reinterpret_cast<const uint4*>(xp);\n                  z1 = *reinterpret_cast<const uint4*>(xp + 8);\n                } else {\n                  z0 = wp_bde7122f9b4f_ldx(xp); z1 = wp_bde7122f9b4f_ldx(xp + (XM >= 9 ? 32 : 8));\n                }\n                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;\n                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;\n              }\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);\n                } else {\n                  wp_bde7122f9b4f_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));\n                  if constexpr (HILO) wp_bde7122f9b4f_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));\n                }\n              }\n            }\n          }\n          if constexpr (XM == 4) __syncthreads();\n        }\n      }\n    }\n\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          wp_bde7122f9b4f_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) wp_bde7122f9b4f_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                wp_bde7122f9b4f_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) wp_bde7122f9b4f_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) wp_bde7122f9b4f_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);\n          const int NI = WR * RT * MTT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            wp_bde7122f9b4f_reduce(part, NI, WK, NI, tid, nt, emit_full);\n          } else {\n            float4* csum = part + (size_t)NW * RT * MTT * 32;\n            wp_bde7122f9b4f_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) wp_bde7122f9b4f_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2 || RED == 4) {\n            wp_bde7122f9b4f_cl_wait();\n            if (RED == 4 || t < TV) {\n              const uint32_t rbar = wp_bde7122f9b4f_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = (tg * MT + n) * 4 + t;\n                  if (RED == 4 || tn < TV) {\n                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else wp_bde7122f9b4f_st_async4(wp_bde7122f9b4f_mapa(wp_bde7122f9b4f_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            wp_bde7122f9b4f_wait_bar(bar);\n            if constexpr (RED == 4) wp_bde7122f9b4f_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);\n            else wp_bde7122f9b4f_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (RED == 4 || t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if ((tg * MT + n) * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            wp_bde7122f9b4f_cl_wait();\n            const uint32_t rbar = wp_bde7122f9b4f_mapa(bar, 0);\n            wp_bde7122f9b4f_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else wp_bde7122f9b4f_st_async4(wp_bde7122f9b4f_mapa(wp_bde7122f9b4f_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            wp_bde7122f9b4f_wait_bar(bar);\n            wp_bde7122f9b4f_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int wp_bde7122f9b4f_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>\nstatic int wp_bde7122f9b4f_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = wp_bde7122f9b4f_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG>;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * TG * 8) return -5;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = std::min(4 * MT * TG, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;\n    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;\n  if (XM == 4) sm += (size_t)WR * WK * RT * DC2Fmt<FMT>::NA * 2048;\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = wp_bde7122f9b4f_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[1];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    lc.numAttrs = 1;\n  }\n  if (query == 3) { int v=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); cudaGetLastError(); return v; }\n  if (query == 4) return (int)sm;\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint wp_bde7122f9b4f_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\ncase 0: return wp_bde7122f9b4f_launch_t<1,2,1,1,3,0,128,0,2304,8192,1,4,8,4,7,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);\ncase 1: return wp_bde7122f9b4f_launch_t<0,2,1,1,3,0,128,0,2304,8192,1,4,8,4,7,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);\n    default: return -99;\n  }\n}\n\n'
tt_v10_warps_W0_CPP = '\n\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint wp_bde7122f9b4f_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n\n'
tt_v10_warps_W0_INSTANCES = [(1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (0, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1)]
tt_v10_warps_WARPS_FAMILY_SPECS = {0: {'name': 'wp_bde7122f9b4f', 'launch': 'wp_bde7122f9b4f_launch', 'family': 'dc5', 'decode': 5}}
tt_v10_warps_WARPS_ROUTES = {('qkv_gate', 16): {'family': 0, 'inst': (1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), 'cfg': (1, 4, 8, 4), 'rows': 2304, 'k': 8192}}
tt_v10_warps_WARPS_PACKERS = {0: (tt_v10_warps_wp0_pack_format, tt_v10_warps_wp0_stream)}
tt_v10_warps_WARPS_SOURCES = {0: (tt_v10_warps_W0_CPP, tt_v10_warps_W0_CUDA, tt_v10_warps_W0_INSTANCES)}

def tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0) -> int:
    return ext[sp['family']](inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), x.shape[0], n, x.shape[1], *cfg, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_warps_warps_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8) -> bool:
    n, k = exact.shape
    x = torch.zeros((rows_per_launch, k), device=exact.device, dtype=torch.bfloat16)
    out = torch.empty((rows_per_launch, n), device=exact.device, dtype=torch.bfloat16)
    for column in range(0, k, rows_per_launch):
        count = min(rows_per_launch, k - column)
        x.zero_()
        x[torch.arange(count, device=x.device), torch.arange(column, column + count, device=x.device)] = 1
        if tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg):
            return False
        if not torch.equal(out[:count], exact[:, column:column + count].t()):
            return False
    return True

def tt_v10_warps_prepare_warps(weights) -> dict:
    state = {'plans': {}, 'ext': {}}
    entries = []
    for (layer, m), route in tt_v10_warps_WARPS_ROUTES.items():
        weight = weights.get(layer)
        if weight is None or weight.dtype != torch.bfloat16 or (not weight.is_cuda):
            continue
        if weight.ndim != 2 or weight.shape[0] < route['rows'] or weight.shape[1] != route['k']:
            continue
        if torch.cuda.get_device_capability(weight.device) != (9, 0):
            continue
        entries.append(((layer, m), route))
    for family in sorted({r['family'] for _, r in entries}):
        spec = tt_v10_warps_WARPS_FAMILY_SPECS[family]
        cpp, cuda, instances = tt_v10_warps_WARPS_SOURCES[family]
        ext = load_inline(spec['name'], cpp, cuda_sources=cuda, functions=[spec['launch']], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        state['ext'][family] = getattr(ext, spec['launch'])
    packs = {}
    streams = {}
    for key, route in entries:
        layer, m = key
        family = route['family']
        spec = tt_v10_warps_WARPS_FAMILY_SPECS[family]
        weight = weights[layer][:route['rows']].contiguous()
        n, k = weight.shape
        pack_format, stream = tt_v10_warps_WARPS_PACKERS[family]
        cfg = route['cfg']
        original = route['inst']
        for fmt in (1, 0):
            inst = (fmt, *original[1:])
            pack_key = (family, layer, fmt, n)
            if pack_key not in packs:
                packs[pack_key] = pack_format(weight, fmt, decode=spec['decode']) if spec['decode'] is not None else pack_format(weight, fmt)
            packed = packs[pack_key]
            if packed is None:
                continue
            stream_key = pack_key + (inst[2], *cfg)
            if stream_key not in streams:
                if spec['family'].startswith('rag_'):
                    rows, columns, size = packed['data'].shape
                    target = cfg[1] * cfg[2] * cfg[3]
                    padded = torch.zeros((rows, target, size), device=weight.device, dtype=packed['data'].dtype)
                    padded[:, :columns].copy_(packed['data'])
                    packed = dict(packed, data=padded)
                streams[stream_key] = stream(packed, inst[2], *cfg)
            sp = streams[stream_key]
            if sp is None:
                continue
            index = tt_v10_warps_WARPS_SOURCES[family][2].index(inst)
            plan = dict(sp, family=family, inst_id=index, cfg=cfg, rows=n, k=k, inst=inst)
            x = torch.empty((m, k), device=weight.device, dtype=torch.bfloat16)
            out = torch.empty((m, n), device=weight.device, dtype=torch.bfloat16)
            attr = tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 2)
            occupancy = tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 1)
            required = n // (16 * inst[2] * cfg[0]) if cfg[3] > 1 else (n // (16 * inst[2] * cfg[0]) + 131) // 132
            if attr < 0 or attr % 1000 or occupancy < required:
                continue
            if not tt_v10_warps_warps_verify_onehot(state['ext'], index, plan, weight, cfg, inst[1] * 8):
                continue
            state['plans'][key] = plan
            break
    return state

def tt_v10_warps_run_warps(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = tt_v10_warps_warps_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None

"""Lossless m32 pooler and qkv projections, with an optional unpooler route."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_m32x_M32X_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_m32x_M32X_ESC_CAP = 32
tt_v10_m32x__CFG = (1, 8, 16, 1)
tt_v10_m32x__CUDA = '#define DC2_DECODE 5\n\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define IFD __device__ __forceinline__\n\nIFD int4 m32x_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nIFD uint4 m32x_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nIFD uint32_t m32x_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nIFD uint32_t m32x_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nIFD void m32x_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nIFD void m32x_dec_old(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = m32x_prmt(T0, T1, cm);\n  const uint32_t hhi = m32x_prmt(T0, T1, cm >> 16);\n  a[0] = m32x_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = m32x_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = m32x_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = m32x_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\nIFD uint32_t m32x_mul(uint32_t a, uint32_t scale) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(scale));\n  return d;\n}\n\nIFD uint32_t m32x_mask_bias(uint32_t a) {\n  uint32_t d;\n  asm("lop3.b32 %0, %1, 0x87ff87ff, 0x40004000, 0xea;" : "=r"(d) : "r"(a));\n  return d;\n}\n\nIFD void m32x_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n#if DC2_DECODE == 0\n  m32x_dec_old(L0, L1, C, T0, T1, a);\n#elif DC2_DECODE == 1\n  m32x_dec_old(L0, L1, C, 0x43424140u, 0x47464544u, a);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) a[q] = m32x_mul(a[q], T0);\n#else\n#if DC2_DECODE >= 4\n  uint32_t hi;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(hi) : "r"(C));\n#else\n  const uint32_t hi = ((C >> 4) & 0x07070707u) | ((C << 4) & 0x80808080u);\n#endif\n  a[0] = m32x_prmt(L0, C, 0x5140u);\n  a[1] = m32x_prmt(L0, C, 0x7362u);\n  a[2] = m32x_prmt(L1, hi, 0x5140u);\n  a[3] = m32x_prmt(L1, hi, 0x7362u);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) {\n#if DC2_DECODE == 2 || DC2_DECODE == 4\n    a[q] = m32x_mul(m32x_mask_bias(a[q]), T0);\n#else\n    a[q] = (a[q] & 0x87ff87ffu) + T0;\n#endif\n  }\n#endif\n}\n\n__device__ __noinline__ void m32x_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nIFD void m32x_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) m32x_never(p);\n}\n\nIFD uint32_t m32x_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nIFD uint32_t m32x_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nIFD uint32_t m32x_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nIFD void m32x_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nIFD void m32x_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nIFD void m32x_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nIFD void m32x_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nIFD void m32x_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nIFD void m32x_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nIFD bool m32x_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nIFD uint64_t m32x_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a wrong result instead of a hang.\nIFD void m32x_wait_bar(uint32_t bar) {\n  if (m32x_mbar_try(bar)) return;\n  const uint64_t t0 = m32x_gtime();\n  while (!m32x_mbar_try(bar))\n    if (m32x_gtime() - t0 > 20000000ull) break;\n}\n\nIFD void m32x_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nIFD float4 m32x_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nIFD void m32x_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nIFD void m32x_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct M32xFmt;\ntemplate <> struct M32xFmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct M32xFmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct M32xFmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct M32xFmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nIFD void m32x_decode(const int4 (&w)[M32xFmt<FMT>::NL], uint32_t (&A)[M32xFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < M32xFmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < M32xFmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      m32x_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      m32x_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      m32x_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      m32x_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nIFD void m32x_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nIFD void m32x_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) m32x_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>\n__global__ void __launch_bounds__(NTMAX) m32x_gemv(\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = M32xFmt<FMT>::NL, NA = M32xFmt<FMT>::NA, MTT = MT * TG;\n  constexpr bool NOW = DIAG & 256;\n  constexpr bool C12 = (FMT == 1 || FMT == 3) && !NOW;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;\n  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);\n  const int rawbytes = CP * WR * WK * RT * NL * 512;\n  const int wbytes = rawbytes + (XM == 4 ? CWR * CWK * RT * NA * 2048 : 0);\n  int4* sw = reinterpret_cast<int4*>(smem + xbytes);\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;\n    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = min(4 * MTT, (m + 1) >> 1);\n    const uint32_t bar = m32x_smem_u32(&s_bar);\n    uint32_t rank = 0;\n    if constexpr (PUSH) {\n      rank = m32x_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        m32x_mbar_init(bar, 1);\n        m32x_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);\n        m32x_fence_init();\n      }\n      m32x_cl_arrive();\n    }\n\n    if constexpr (XM == 2 || XM == 3 || XM == 5) {\n      const int ks = K / S;\n      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {\n        const int t = v / (ks / 8), c = v % (ks / 8);\n        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;\n        void* dst = sx + t * (ks + 8) + c * 8;\n        if constexpr (XM == 3 || XM == 5) {\n          const uint32_t sa = m32x_smem_u32(dst);\n          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");\n        } else *reinterpret_cast<uint4*>(dst) = m32x_ldx(src);\n      }\n      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");\n      else __syncthreads();\n    }\n    uint32_t my_e = 0xFFFFFFFFu;\n    if constexpr (C12 && !RO) my_e = m32x_lde(esc + (size_t)gw * 32 + lane);\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);\n\n    int4* rawring = reinterpret_cast<int4*>(smem + xbytes);\n    auto stage_weights = [&](int slot, int si) {\n      if constexpr (CP > 0 && !NOW) {\n        if (si < KU) {\n#pragma unroll\n          for (int j=0; j<RT; ++j)\n#pragma unroll\n            for (int l=0; l<NL; ++l) {\n              const uint32_t dst=m32x_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);\n              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;\n              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");\n            }\n        }\n        asm volatile("cp.async.commit_group;" ::: "memory");\n      }\n    };\n#pragma unroll\n    for (int q=0; q<CP; ++q) stage_weights(q,q);\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) if constexpr (CP == 0 && !NOW) if (XM != 4 || tg == 0) w[slot][j][l] = m32x_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = m32x_ldx(xp);\n          xv[slot][n][1] = m32x_ldx(xp + (XM >= 9 ? 32 : 8));\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    m32x_order_point(m, outp);\n    if constexpr (XM == 3 || XM == 5) {\n      asm volatile("cp.async.wait_group 0;" ::: "memory");\n      __syncthreads();\n    }\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          m32x_order_point(m, outp);\n        } else {\n          uint32_t b[MT][8];\n          if constexpr (!NOX && (XM == 7 || XM == 8 || XM == 11)) {\n#pragma unroll\n            for (int n = 0; n < MT; ++n) {\n              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;\n              const uint4 z0=m32x_ldx(xp), z1=m32x_ldx(xp+(XM >= 9 ? 32 : 8));\n              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;\n              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;\n            }\n            m32x_order_point(m, outp);\n          }\n          if constexpr (CP > 0 && !NOW) {\n            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");\n#pragma unroll\n            for (int j=0; j<RT; ++j)\n#pragma unroll\n              for (int l=0; l<NL; ++l) {\n                const uint32_t src=m32x_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);\n                int4& v=w[s][j][l];\n                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");\n              }\n          }\n          if constexpr (CP > 0 && !NOW) stage_weights(si%CP,si+CP);\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) {\n            if constexpr (NOW) {\n#pragma unroll\n              for (int a=0;a<NA*16;a++) A[j][a]=0x3f803f80u;\n            } else if (XM != 4 || tg == 0) m32x_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n          }\n\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          if (si + ST < KU) load(s, si + ST);\n          m32x_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) m32x_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n          if constexpr (XM == 4) {\n            if (tg == 0) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int q = 0; q < NA * 4; ++q)\n                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =\n                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);\n            }\n            __syncthreads();\n            if (tg != 0) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int q = 0; q < NA * 4; ++q) {\n                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];\n                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;\n                }\n            }\n          }\n          if constexpr (XM == 6 || XM == 8 || XM == 10) {\n#pragma unroll\n            for (int q = 0; q < 4; ++q)\n#pragma unroll\n              for (int n = 0; n < MT; ++n)\n#pragma unroll\n                for (int j = 0; j < RT; ++j)\n                  m32x_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);\n          } else if constexpr (XM == 5) {\n#pragma unroll\n            for (int q = 0; q < 4; ++q) {\n#pragma unroll\n              for (int np = 0; np < (MT + 1) / 2; ++np) {\n                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);\n                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;\n                const uint32_t addr = m32x_smem_u32(sx + tok * (K / S + 8) + col);\n                uint32_t b0,b1,b2,b3;\n                if constexpr (MT == 1)\n                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));\n                else\n                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));\n#pragma unroll\n                for (int j = 0; j < RT; ++j) {\n                  m32x_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);\n                  if constexpr (MT > 1) m32x_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);\n                }\n              }\n            }\n          } else {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n) {\n              uint32_t bx[8];\n              if constexpr (XM >= 1 && XM <= 4 && !NOX) {\n                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);\n                const __nv_bfloat16* xp;\n                if constexpr (XM == 2 || XM == 3 || XM == 5)\n                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;\n                else xp = x0 + (size_t)tok * K + si * 64;\n                uint4 z0, z1;\n                if constexpr (XM == 2 || XM == 3 || XM == 5) {\n                  z0 = *reinterpret_cast<const uint4*>(xp);\n                  z1 = *reinterpret_cast<const uint4*>(xp + 8);\n                } else {\n                  z0 = m32x_ldx(xp); z1 = m32x_ldx(xp + (XM >= 9 ? 32 : 8));\n                }\n                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;\n                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;\n              }\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);\n                } else {\n                  m32x_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));\n                  if constexpr (HILO) m32x_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));\n                }\n              }\n            }\n          }\n          if constexpr (XM == 4) __syncthreads();\n        }\n      }\n    }\n\n    if constexpr (CP > 0 && !NOW) asm volatile("cp.async.wait_group 0;" ::: "memory");\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          m32x_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) m32x_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                m32x_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) m32x_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) m32x_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);\n          const int NI = WR * RT * MTT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            m32x_reduce(part, NI, WK, NI, tid, nt, emit_full);\n          } else {\n            float4* csum = part + (size_t)NW * RT * MTT * 32;\n            m32x_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) m32x_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2 || RED == 4) {\n            m32x_cl_wait();\n            if (RED == 4 || t < TV) {\n              const uint32_t rbar = m32x_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = (tg * MT + n) * 4 + t;\n                  if (RED == 4 || tn < TV) {\n                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else m32x_st_async4(m32x_mapa(m32x_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            m32x_wait_bar(bar);\n            if constexpr (RED == 4) m32x_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);\n            else m32x_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (RED == 4 || t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if ((tg * MT + n) * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            m32x_cl_wait();\n            const uint32_t rbar = m32x_mapa(bar, 0);\n            m32x_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else m32x_st_async4(m32x_mapa(m32x_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            m32x_wait_bar(bar);\n            m32x_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int m32x_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>\nstatic int m32x_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = m32x_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG, CP>;\n  const int policy = query / 100;\n  query %= 100;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * TG * 8) return -5;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = std::min(4 * MT * TG, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;\n    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;\n  if (XM == 4) sm += (size_t)WR * WK * RT * M32xFmt<FMT>::NA * 2048;\n  sm += (size_t)CP * NW * RT * M32xFmt<FMT>::NL * 512;\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = m32x_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[2];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;\n    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);\n    lc.numAttrs = 2;\n  }\n  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }\n  if (query == 4) return (int)sm;\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\ncase 0:return m32x_launch_t<1,4,1,1,0,0,256,0,4096,8192,1,8,16,1,7,1,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);\ncase 1:return m32x_launch_t<0,4,1,1,0,0,256,0,4096,8192,1,8,16,1,7,1,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);\n    default: return -99;\n  }\n}\n'
tt_v10_m32x__CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def tt_v10_m32x_m32x_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_m32x_m32x_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_m32x_m32x_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_m32x_m32x_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_m32x_m32x_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
    """Inverse of m32x_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_v10_m32x_m32x_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v10_m32x_m32x_units(hi), tt_v10_m32x_m32x_units(lo)]
    else:
        mats = [tt_v10_m32x_m32x_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v10_m32x_m32x_c12_bytes(v, decode)
        if not torch.equal(tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(tt_v10_m32x_m32x_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v10_m32x_M32X_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v10_m32x_M32X_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v10_m32x_m32x_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32x_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_m32x_m32x_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_m32x_m32x_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v10_m32x__build():
    name = 'm32x_pooler_' + hashlib.sha256(tt_v10_m32x__CUDA.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_m32x__CPP, cuda_sources=tt_v10_m32x__CUDA, functions=['m32x_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__make_plan(ext, weights, fmt):
    pack = tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = tt_v10_m32x_m32x_stream(pack, 1, *tt_v10_m32x__CFG)
    if stream is None:
        return None
    return dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=4096, cfg=tt_v10_m32x__CFG)

def tt_v10_m32x__run(x, plan):
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    rc = tt_v10_m32x_m32x_launch_raw(plan['ext'], plan['idx'], plan, x, out, 4096, tt_v10_m32x__CFG)
    if rc:
        raise RuntimeError(f'm32x launch failed: {rc}')
    return out
tt_v10_m32x__W_CUDA = '#define DC2_DECODE 5\n\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t m32x_w_addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void m32x_w_waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void m32x_w_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void m32x_w_cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t m32x_w_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void m32x_w_tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 m32x_w_ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t m32x_w_prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void m32x_w_dec_old(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = m32x_w_prmt(T0, T1, cm);\n  const uint32_t hhi = m32x_w_prmt(T0, T1, cm >> 16);\n  a[0] = m32x_w_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = m32x_w_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = m32x_w_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = m32x_w_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\nD uint32_t m32x_w_mul(uint32_t a, uint32_t scale) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(scale));\n  return d;\n}\n\nD uint32_t m32x_w_mask_bias(uint32_t a) {\n  uint32_t d;\n  asm("lop3.b32 %0, %1, 0x87ff87ff, 0x40004000, 0xea;" : "=r"(d) : "r"(a));\n  return d;\n}\n\nD void m32x_w_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n#if DC2_DECODE == 0\n  m32x_w_dec_old(L0, L1, C, T0, T1, a);\n#elif DC2_DECODE == 1\n  m32x_w_dec_old(L0, L1, C, 0x43424140u, 0x47464544u, a);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) a[q] = m32x_w_mul(a[q], T0);\n#else\n#if DC2_DECODE >= 4\n  uint32_t hi;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(hi) : "r"(C));\n#else\n  const uint32_t hi = ((C >> 4) & 0x07070707u) | ((C << 4) & 0x80808080u);\n#endif\n  a[0] = m32x_w_prmt(L0, C, 0x5140u);\n  a[1] = m32x_w_prmt(L0, C, 0x7362u);\n  a[2] = m32x_w_prmt(L1, hi, 0x5140u);\n  a[3] = m32x_w_prmt(L1, hi, 0x7362u);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) {\n#if DC2_DECODE == 2 || DC2_DECODE == 4\n    a[q] = m32x_w_mul(m32x_w_mask_bias(a[q]), T0);\n#else\n    a[q] = (a[q] & 0x87ff87ffu) + T0;\n#endif\n  }\n#endif\n}\n\nD void m32x_w_patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\nD void m32x_w_mma16(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %24, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%12,%13,%14,%15}, %25, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, %26, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21,%22,%23}, %27, 1, 1, 1, 0; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]),"l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}\nD void m32x_w_mma32(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %32, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%20,%21,%22,%23}, %33, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%24,%25,%26,%27}, %34, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%28,%29,%30,%31}, %35, 1, 1, 1, 0; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]),"l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}\nD void m32x_w_mma64(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %48, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%36,%37,%38,%39}, %49, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%40,%41,%42,%43}, %50, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%44,%45,%46,%47}, %51, 1, 1, 1, 0; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]),"l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}\n\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\n__global__ __launch_bounds__(128*WK,CAP) void m32x_mws_a5a0ae72b156_gemm(\n    const __grid_constant__ CUtensorMap tx,const uint4* __restrict__ w,\n    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {\n  extern __shared__ __align__(1024) uint8_t sm[];\n  constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;\n  int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;\n  uint32_t base0=m32x_w_addr(sm),base=base0+wg*WS,full=base0+WK*WS+wg*Q*8;\n  int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;\n  int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;\n  uint64_t pw,px;\n  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));\n  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(px));\n  if(tid%128==0) {\n    for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  auto loadx=[&](int i) {\n    if(tid%128==0 && i<nkb) {\n      int s=i%Q;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n      m32x_w_tma(base+s*SB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<Q;s++)loadx(s);\n  uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;\n  if(C12 && warp<RW) {\n    const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;\n    e0=el[lane];e1=el[lane+32];nxt=__shfl_sync(0xffffffffu,e0,0);\n  }\n  uint4 raw[PF][NL]={};\n  auto load=[&](int slot,int i) {\n    if(i<nkb && warp<RW) {\n      const uint4* src=w+((long)(nt*(k/64)+kb0+i)*RW+warp)*NL*32+lane;\n      #pragma unroll\n      for(int l=0;l<NL;l++)raw[slot][l]=m32x_w_ldw(src+l*32,pw);\n    }\n  };\n  #pragma unroll\n  for(int p=0;p<PF;p++)load(p,p);\n  float acc[BN/2];\n  #pragma unroll\n  for(int j=0;j<BN/2;j++)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));\n  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {\n    #pragma unroll\n    for(int p=0;p<PF;p++) {\n      int i=ib+p;if(i>=nkb)break;\n      uint32_t a[16];\n      if constexpr(C12) {\n        m32x_w_dec(raw[p][0].x,raw[p][0].y,raw[p][2].x,t0,t1,a);\n        m32x_w_dec(raw[p][0].z,raw[p][0].w,raw[p][2].y,t0,t1,a+4);\n        m32x_w_dec(raw[p][1].x,raw[p][1].y,raw[p][2].z,t0,t1,a+8);\n        m32x_w_dec(raw[p][1].z,raw[p][1].w,raw[p][2].w,t0,t1,a+12);\n      } else {\n        #pragma unroll\n        for(int l=0;l<4;l++) {\n          a[4*l]=raw[p][l].x;a[4*l+1]=raw[p][l].y;a[4*l+2]=raw[p][l].z;a[4*l+3]=raw[p][l].w;\n        }\n      }\n      load(p,i+PF);\n      if constexpr(C12) {\n        while((nxt>>18)==uint32_t(i)) {\n          if(lane==int((nxt>>13)&31u))m32x_w_patch(a,nxt);\n          ++ep;\n          uint32_t ev=ep<32?e0:e1;\n          nxt=__shfl_sync(0xffffffffu,ev,ep&31);\n          if(ep>=64)nxt=0xffffffffu;\n        }\n      }\n      int s=i%Q;\n      m32x_w_waitbar(full+s*8,(i/Q)&1);\n      #pragma unroll\n      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");\n      if constexpr(BN==16)m32x_w_mma16(acc,a,m32x_w_desc(base+s*SB));\n      if constexpr(BN==32)m32x_w_mma32(acc,a,m32x_w_desc(base+s*SB));\n      if constexpr(BN==64)m32x_w_mma64(acc,a,m32x_w_desc(base+s*SB));\n      #pragma unroll\n      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");\n      loadx(i+Q);\n    }\n  }\n  __syncthreads();\n  int row=warp*16+lane/4,col=2*(lane%4);\n  #pragma unroll\n  for(int j=0;j<BN/8;j++) {\n    #pragma unroll\n    for(int h=0;h<2;h++) {\n      if(warp>=RW)continue;\n      ((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];\n      ((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];\n    }\n  }\n  __syncthreads();\n  if(split>1)m32x_w_cluster_sync();\n  for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split) {\n    int token=q/(R/4),row=(q%(R/4))*4;\n    if(mt*BN+token>=m)continue;\n    float4 sum=make_float4(0,0,0,0);\n    for(int pp=0;pp<split*WK;pp++) {\n      int s=pp/WK;\n      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;\n      if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));\n      float4 v;\n      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(remote):"memory");\n      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;\n    }\n    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);\n    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);\n  }\n  if(split>1)m32x_w_cluster_sync();\n}\n\n\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint m32x_w_map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint m32x_mws_a5a0ae72b156_launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=m32x_w_map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=(a>b?a:b)+WK*Q*8;\n  auto kern=m32x_mws_a5a0ae72b156_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint m32x_w_launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {case 0: return m32x_mws_a5a0ae72b156_launch_t<32,64,4,2,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);\ncase 1: return m32x_mws_a5a0ae72b156_launch_t<32,64,4,2,2,false,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);}\n  return -1;\n}\nint m32x_w_info(int id){cudaFuncAttributes fa={};switch(id){case 0:cudaFuncGetAttributes(&fa,m32x_mws_a5a0ae72b156_gemm<32,64,4,2,2,true,1>);break;\ncase 1:cudaFuncGetAttributes(&fa,m32x_mws_a5a0ae72b156_gemm<32,64,4,2,2,false,1>);break;}return fa.numRegs*10000+fa.localSizeBytes;}\n'
tt_v10_m32x__W_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_w_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint m32x_w_info(int);\n'

def tt_v10_m32x__w_c12_bytes(v):
    return tt_v10_m32x_m32x_c12_bytes(v, 5)

def tt_v10_m32x__w_c12_unpack(data, base, esc):
    return tt_v10_m32x_m32x_c12_unpack(data, base, esc, 5)

def tt_v10_m32x__w_table(base):
    return base << 8 | base << 24

def tt_v10_m32x__w_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = tt_v10_m32x__w_c12_bytes(v)
        if not torch.equal(tt_v10_m32x__w_c12_unpack(data, b, entries), v):
            return None
        table = tt_v10_m32x__w_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = tt_v10_m32x_m32x_to_i32(entry)
    else:
        data = tt_v10_m32x_m32x_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def tt_v10_m32x__w_launch_raw(ext, idx, pk, x, out):
    return ext.m32x_w_launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def tt_v10_m32x__w_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if tt_v10_m32x__w_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))

def tt_v10_m32x__w_build():
    name = 'm32x_unpooler_' + hashlib.sha256(tt_v10_m32x__W_CUDA.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_m32x__W_CPP, cuda_sources=tt_v10_m32x__W_CUDA, functions=['m32x_w_launch', 'm32x_w_info'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__w_make_plan(ext, weights, fmt):
    pack = tt_v10_m32x__w_pack_weight(weights, 64, 4, 1, fmt == 1)
    if pack is None:
        return None
    return dict(pack, ext=ext, idx=0 if fmt == 1 else 1, fmt=fmt)

def tt_v10_m32x__w_run(x, plan):
    out = torch.empty((x.shape[0], 8192), dtype=torch.bfloat16, device=x.device)
    rc = tt_v10_m32x__w_launch_raw(plan['ext'], plan['idx'], plan, x, out)
    if rc:
        raise RuntimeError(f'm32x unpooler launch failed: {rc}')
    return out

def tt_v10_m32x__w_verify_random(plan, weights):
    for m in (9, 16, 17, 24, 32):
        x = torch.randn((m, 4096), device=weights.device, dtype=torch.bfloat16)
        x[-1].zero_()
        x[-1, m * 131 % 4096] = 16
        ref = (x.double() @ weights.double().T).to(torch.bfloat16).double()
        out = tt_v10_m32x__w_run(x, plan).double()
        diff = out - ref
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 0.0025:
            return False
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if float(rows.max()) > 0.01 or float(rows[-1]) > 0.00025:
            return False
    return True
tt_v10_m32x__Q_CFG = (1, 4, 8, 4)
tt_v10_m32x__Q_CUDA = '#define DC2_DECODE 5\n\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define IFD __device__ __forceinline__\n\nIFD int4 m32x_q_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nIFD uint4 m32x_q_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nIFD uint32_t m32x_q_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nIFD uint32_t m32x_q_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nIFD void m32x_q_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nIFD void m32x_q_dec_old(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = m32x_q_prmt(T0, T1, cm);\n  const uint32_t hhi = m32x_q_prmt(T0, T1, cm >> 16);\n  a[0] = m32x_q_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = m32x_q_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = m32x_q_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = m32x_q_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\nIFD uint32_t m32x_q_mul(uint32_t a, uint32_t scale) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(scale));\n  return d;\n}\n\nIFD uint32_t m32x_q_mask_bias(uint32_t a) {\n  uint32_t d;\n  asm("lop3.b32 %0, %1, 0x87ff87ff, 0x40004000, 0xea;" : "=r"(d) : "r"(a));\n  return d;\n}\n\nIFD void m32x_q_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n#if DC2_DECODE == 0\n  m32x_q_dec_old(L0, L1, C, T0, T1, a);\n#elif DC2_DECODE == 1\n  m32x_q_dec_old(L0, L1, C, 0x43424140u, 0x47464544u, a);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) a[q] = m32x_q_mul(a[q], T0);\n#else\n#if DC2_DECODE >= 4\n  uint32_t hi;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(hi) : "r"(C));\n#else\n  const uint32_t hi = ((C >> 4) & 0x07070707u) | ((C << 4) & 0x80808080u);\n#endif\n  a[0] = m32x_q_prmt(L0, C, 0x5140u);\n  a[1] = m32x_q_prmt(L0, C, 0x7362u);\n  a[2] = m32x_q_prmt(L1, hi, 0x5140u);\n  a[3] = m32x_q_prmt(L1, hi, 0x7362u);\n#pragma unroll\n  for (int q = 0; q < 4; ++q) {\n#if DC2_DECODE == 2 || DC2_DECODE == 4\n    a[q] = m32x_q_mul(m32x_q_mask_bias(a[q]), T0);\n#else\n    a[q] = (a[q] & 0x87ff87ffu) + T0;\n#endif\n  }\n#endif\n}\n\n__device__ __noinline__ void m32x_q_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nIFD void m32x_q_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) m32x_q_never(p);\n}\n\nIFD uint32_t m32x_q_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nIFD uint32_t m32x_q_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nIFD uint32_t m32x_q_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nIFD void m32x_q_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nIFD void m32x_q_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nIFD void m32x_q_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nIFD void m32x_q_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nIFD void m32x_q_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nIFD void m32x_q_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nIFD bool m32x_q_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nIFD uint64_t m32x_q_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a wrong result instead of a hang.\nIFD void m32x_q_wait_bar(uint32_t bar) {\n  if (m32x_q_mbar_try(bar)) return;\n  const uint64_t t0 = m32x_q_gtime();\n  while (!m32x_q_mbar_try(bar))\n    if (m32x_q_gtime() - t0 > 20000000ull) break;\n}\n\nIFD void m32x_q_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nIFD float4 m32x_q_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nIFD void m32x_q_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nIFD void m32x_q_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct M32xQFmt;\ntemplate <> struct M32xQFmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct M32xQFmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct M32xQFmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct M32xQFmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nIFD void m32x_q_decode(const int4 (&w)[M32xQFmt<FMT>::NL], uint32_t (&A)[M32xQFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < M32xQFmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < M32xQFmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      m32x_q_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      m32x_q_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      m32x_q_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      m32x_q_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nIFD void m32x_q_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nIFD void m32x_q_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) m32x_q_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>\n__global__ void __launch_bounds__(NTMAX,(NTMAX==128?5:3)) m32x_q_gemv(\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = M32xQFmt<FMT>::NL, NA = M32xQFmt<FMT>::NA, MTT = MT * TG;\n  constexpr bool NOW = DIAG & 256;\n  constexpr bool C12 = (FMT == 1 || FMT == 3) && !NOW;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED >= 2 && RED <= 6) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;\n  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);\n  const int rawbytes = CP * WR * WK * RT * NL * 512;\n  const int wbytes = rawbytes + (XM == 4 ? CWR * CWK * RT * NA * 2048 : 0);\n  int4* sw = reinterpret_cast<int4*>(smem + xbytes);\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;\n    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = min(4 * MTT, (m + 1) >> 1);\n    const uint32_t bar = m32x_q_smem_u32(&s_bar);\n    uint32_t rank = 0;\n    if constexpr (PUSH) {\n      rank = m32x_q_rank();\n      if constexpr(RED >= 5) {\n        if(tid == 0) m32x_q_mbar_init(bar, 1);\n        __syncthreads();\n      }\n      if ((RED >= 5 || rank == 0) && tid == 0) {\n        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        if constexpr(RED < 5) m32x_q_mbar_init(bar, 1);\n        if constexpr(RED >= 5) {\n          m32x_q_mbar_expect(bar, (S-1)*(RED==5?NW:WR)*RT*(MTT/S)*512u);\n        } else m32x_q_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);\n        m32x_q_fence_init();\n      }\n      m32x_q_cl_arrive();\n    }\n\n    if constexpr (XM == 2 || XM == 3 || XM == 5) {\n      const int ks = K / S;\n      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {\n        const int t = v / (ks / 8), c = v % (ks / 8);\n        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;\n        void* dst = sx + t * (ks + 8) + c * 8;\n        if constexpr (XM == 3 || XM == 5) {\n          const uint32_t sa = m32x_q_smem_u32(dst);\n          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");\n        } else *reinterpret_cast<uint4*>(dst) = m32x_q_ldx(src);\n      }\n      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");\n      else __syncthreads();\n    }\n    uint32_t my_e = 0xFFFFFFFFu;\n    if constexpr (C12 && !RO) my_e = m32x_q_lde(esc + (size_t)gw * 32 + lane);\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);\n\n    int4* rawring = reinterpret_cast<int4*>(smem + xbytes);\n    auto stage_weights = [&](int slot, int si) {\n      if constexpr (CP > 0 && !NOW) {\n        if (si < KU) {\n#pragma unroll\n          for (int j=0; j<RT; ++j)\n#pragma unroll\n            for (int l=0; l<NL; ++l) {\n              const uint32_t dst=m32x_q_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);\n              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;\n              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");\n            }\n        }\n        asm volatile("cp.async.commit_group;" ::: "memory");\n      }\n    };\n#pragma unroll\n    for (int q=0; q<CP; ++q) stage_weights(q,q);\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) if constexpr (CP == 0 && !NOW) if (XM != 4 || tg == 0) w[slot][j][l] = m32x_q_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = m32x_q_ldx(xp);\n          xv[slot][n][1] = m32x_q_ldx(xp + (XM >= 9 ? 32 : 8));\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    m32x_q_order_point(m, outp);\n    if constexpr (XM == 3 || XM == 5) {\n      asm volatile("cp.async.wait_group 0;" ::: "memory");\n      __syncthreads();\n    }\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          m32x_q_order_point(m, outp);\n        } else {\n          uint32_t b[MT][8];\n          if constexpr (!NOX && (XM == 7 || XM == 8 || XM == 11)) {\n#pragma unroll\n            for (int n = 0; n < MT; ++n) {\n              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;\n              const uint4 z0=m32x_q_ldx(xp), z1=m32x_q_ldx(xp+(XM >= 9 ? 32 : 8));\n              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;\n              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;\n            }\n            m32x_q_order_point(m, outp);\n          }\n          if constexpr (CP > 0 && !NOW) {\n            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");\n#pragma unroll\n            for (int j=0; j<RT; ++j)\n#pragma unroll\n              for (int l=0; l<NL; ++l) {\n                const uint32_t src=m32x_q_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);\n                int4& v=w[s][j][l];\n                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");\n              }\n          }\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) {\n            if constexpr (NOW) {\n#pragma unroll\n              for (int a=0;a<NA*16;a++) A[j][a]=0x3f803f80u;\n            } else if (XM != 4 || tg == 0) m32x_q_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n          }\n          if constexpr (CP > 0 && !NOW) stage_weights(si%CP,si+CP);\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          if (si + ST < KU) load(s, si + ST);\n          m32x_q_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) m32x_q_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n          if constexpr (XM == 4) {\n            if (tg == 0) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int q = 0; q < NA * 4; ++q)\n                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =\n                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);\n            }\n            __syncthreads();\n            if (tg != 0) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int q = 0; q < NA * 4; ++q) {\n                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];\n                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;\n                }\n            }\n          }\n          if constexpr (XM == 6 || XM == 8 || XM == 10) {\n#pragma unroll\n            for (int q = 0; q < 4; ++q)\n#pragma unroll\n              for (int n = 0; n < MT; ++n)\n#pragma unroll\n                for (int j = 0; j < RT; ++j)\n                  m32x_q_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);\n          } else if constexpr (XM == 5) {\n#pragma unroll\n            for (int q = 0; q < 4; ++q) {\n#pragma unroll\n              for (int np = 0; np < (MT + 1) / 2; ++np) {\n                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);\n                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;\n                const uint32_t addr = m32x_q_smem_u32(sx + tok * (K / S + 8) + col);\n                uint32_t b0,b1,b2,b3;\n                if constexpr (MT == 1)\n                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));\n                else\n                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));\n#pragma unroll\n                for (int j = 0; j < RT; ++j) {\n                  m32x_q_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);\n                  if constexpr (MT > 1) m32x_q_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);\n                }\n              }\n            }\n          } else {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n) {\n              uint32_t bx[8];\n              if constexpr (XM >= 1 && XM <= 4 && !NOX) {\n                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);\n                const __nv_bfloat16* xp;\n                if constexpr (XM == 2 || XM == 3 || XM == 5)\n                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;\n                else xp = x0 + (size_t)tok * K + si * 64;\n                uint4 z0, z1;\n                if constexpr (XM == 2 || XM == 3 || XM == 5) {\n                  z0 = *reinterpret_cast<const uint4*>(xp);\n                  z1 = *reinterpret_cast<const uint4*>(xp + 8);\n                } else {\n                  z0 = m32x_q_ldx(xp); z1 = m32x_q_ldx(xp + (XM >= 9 ? 32 : 8));\n                }\n                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;\n                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;\n              }\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);\n                } else {\n                  m32x_q_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));\n                  if constexpr (HILO) m32x_q_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));\n                }\n              }\n            }\n          }\n          if constexpr (XM == 4) __syncthreads();\n        }\n      }\n    }\n\n    if constexpr (CP > 0 && !NOW) asm volatile("cp.async.wait_group 0;" ::: "memory");\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          m32x_q_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) m32x_q_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                m32x_q_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) m32x_q_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) m32x_q_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED >= 5) {\n          const int NI = WR*RT*MTT*32;\n          const int OWN = NI/S;\n          float4* recv = reinterpret_cast<float4*>(smem+xbytes+wbytes);\n          auto emit_owned = [&](int idx, float4 v) {\n            const int ln=idx&31, rest=idx>>5;\n            const int local_n=rest%(MTT/S), tile=rest/(MTT/S);\n            emit_full((tile*MTT+local_n*S+rank)*32+ln,v);\n          };\n          m32x_q_cl_wait();\n          if constexpr(RED==5) {\n#pragma unroll\n            for(int j=0;j<RT;j++)\n#pragma unroll\n              for(int n=0;n<MT;n++) {\n                const int token_tile=tg*MT+n, dst_rank=token_tile%S;\n                const int idx=((wr*RT+j)*(MTT/S)+token_tile/S)*32+lane;\n                float4* dst=recv+(rank*WK+wk)*OWN+idx;\n                const float4 v=make_float4(r[j][n][0],r[j][n][1],r[j][n][2],r[j][n][3]);\n                if(rank==dst_rank) *dst=v;\n                else m32x_q_st_async4(m32x_q_mapa(m32x_q_smem_u32(dst),dst_rank),v,m32x_q_mapa(bar,dst_rank));\n              }\n          } else {\n            float4* part=recv+S*OWN;\n#pragma unroll\n            for(int j=0;j<RT;j++)\n#pragma unroll\n              for(int n=0;n<MT;n++)\n                part[wk*NI+((wr*RT+j)*MTT+tg*MT+n)*32+lane]=make_float4(r[j][n][0],r[j][n][1],r[j][n][2],r[j][n][3]);\n            __syncthreads();\n            m32x_q_reduce(part,NI,WK,NI,tid,nt,[&](int idx,float4 v) {\n              const int ln=idx&31,rest=idx>>5,token_tile=rest%MTT,tile=rest/MTT;\n              const int dst_rank=token_tile%S;\n              const int owned_idx=(tile*(MTT/S)+token_tile/S)*32+ln;\n              float4* dst=recv+rank*OWN+owned_idx;\n              if(rank==dst_rank) *dst=v;\n              else m32x_q_st_async4(m32x_q_mapa(m32x_q_smem_u32(dst),dst_rank),v,m32x_q_mapa(bar,dst_rank));\n            });\n          }\n          __syncthreads();\n          m32x_q_wait_bar(bar);\n          m32x_q_reduce(recv,OWN,S*(RED==5?WK:1),OWN,tid,nt,emit_owned);\n        } else if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);\n          const int NI = WR * RT * MTT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            m32x_q_reduce(part, NI, WK, NI, tid, nt, emit_full);\n          } else {\n            float4* csum = part + (size_t)NW * RT * MTT * 32;\n            m32x_q_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) m32x_q_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2 || RED == 4) {\n            m32x_q_cl_wait();\n            if (RED == 4 || t < TV) {\n              const uint32_t rbar = m32x_q_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = (tg * MT + n) * 4 + t;\n                  if (RED == 4 || tn < TV) {\n                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else m32x_q_st_async4(m32x_q_mapa(m32x_q_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            m32x_q_wait_bar(bar);\n            if constexpr (RED == 4) m32x_q_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);\n            else m32x_q_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (RED == 4 || t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if ((tg * MT + n) * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            m32x_q_cl_wait();\n            const uint32_t rbar = m32x_q_mapa(bar, 0);\n            m32x_q_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else m32x_q_st_async4(m32x_q_mapa(m32x_q_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            m32x_q_wait_bar(bar);\n            m32x_q_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int m32x_q_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>\nstatic int m32x_q_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = m32x_q_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG, CP>;\n  const int policy = query / 100;\n  query %= 100;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * TG * 8) return -5;\n  if (RED>=5 && (MT*TG)%S) return -14;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = std::min(4 * MT * TG, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;\n    else if (RED == 5) sm = (size_t)NW*RT*MT*TG*512;\n    else if (RED == 6) sm = (size_t)(1+WK)*WR*RT*MT*TG*512;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;\n    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;\n  if (XM == 4) sm += (size_t)WR * WK * RT * M32xQFmt<FMT>::NA * 2048;\n  sm += (size_t)CP * NW * RT * M32xQFmt<FMT>::NL * 512;\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = m32x_q_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[2];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;\n    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);\n    lc.numAttrs = 2;\n  }\n  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }\n  if (query == 4) return (int)sm;\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\ncase 0:return m32x_q_launch_t<1,4,1,1,5,0,128,0,2304,8192,1,4,8,4,7,1,0>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);\ncase 1:return m32x_q_launch_t<0,4,1,1,5,0,128,0,2304,8192,1,4,8,4,7,1,0>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);\n    default: return -99;\n  }\n}\n'
tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v10_m32x__q_build():
    name = 'm32x_qkv_' + hashlib.sha256(tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_m32x__Q_CPP, cuda_sources=tt_v10_m32x__Q_CUDA, functions=['m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__q_make_plan(ext, weights, fmt):
    pack = tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = tt_v10_m32x_m32x_stream(pack, 1, *tt_v10_m32x__Q_CFG)
    if stream is None:
        return None
    plan = dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=2304, cfg=tt_v10_m32x__Q_CFG)
    x = weights[:32]
    out = torch.empty((32, 2304), dtype=torch.bfloat16, device=weights.device)
    capacity = tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG, 1)
    attr = tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG, 2)
    if capacity < 144 or attr < 0 or attr % 1000:
        return None
    return plan

def tt_v10_m32x__q_run(x, plan):
    out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
    rc = tt_v10_m32x__q_launch_raw(plan['ext'], plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG)
    if rc:
        raise RuntimeError(f'm32x qkv launch failed: {rc}')
    return out

def tt_v10_m32x_prepare_m32x(weights, *, include_unpooler: bool=False) -> dict:
    state = {'plans': {}, 'device': None}
    configs = [('pooler', (4096, 8192), tt_v10_m32x__build, tt_v10_m32x__make_plan), ('qkv_gate', (2304, 8192), tt_v10_m32x__q_build, tt_v10_m32x__q_make_plan), ('unpooler', (8192, 4096), tt_v10_m32x__w_build, tt_v10_m32x__w_make_plan)]
    for layer, shape, build, make_plan in configs:
        if layer == 'unpooler' and (not include_unpooler):
            continue
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != shape or (not w.is_cuda):
            continue
        if torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        if state['device'] is not None and state['device'] != w.device:
            continue
        state['device'] = w.device
        ext = build()
        for fmt in (1, 0):
            plan = make_plan(ext, w, fmt)
            if plan is None:
                continue
            if layer == 'pooler':
                ok = tt_v10_m32x_m32x_verify_onehot(ext, plan['idx'], plan, w, tt_v10_m32x__CFG, 32) and tt_v10_m32x_m32x_verify_random(ext, plan['idx'], plan, w, tt_v10_m32x__CFG, [9, 16, 17, 24, 32])
            elif layer == 'qkv_gate':
                ok = tt_v10_m32x__q_verify_onehot(ext, plan['idx'], plan, w, tt_v10_m32x__Q_CFG, 32) and tt_v10_m32x__q_verify_random(ext, plan['idx'], plan, w, tt_v10_m32x__Q_CFG, [9, 16, 17, 24, 32])
            else:
                ok = tt_v10_m32x__w_verify_onehot(ext, plan['idx'], plan, w, 32) and tt_v10_m32x__w_verify_random(plan, w)
            if ok:
                state['plans'][layer] = plan
                break
    return state

def tt_v10_m32x_run_m32x(x, layer, n, st) -> torch.Tensor | None:
    expected = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'unpooler': (8192, 4096)}.get(layer)
    if expected is None or x.ndim != 2 or x.shape[0] != 32:
        return None
    if (n, x.shape[1]) != expected or x.dtype != torch.bfloat16 or (not x.is_cuda):
        return None
    if x.device != st.get('device') or not x.is_contiguous():
        return None
    plan = st.get('plans', {}).get(layer)
    if plan is None:
        return None
    if layer == 'pooler':
        return tt_v10_m32x__run(x, plan)
    if layer == 'qkv_gate':
        return tt_v10_m32x__q_run(x, plan)
    return tt_v10_m32x__w_run(x, plan)

V10_ROUTES = [{'m': 8, 'layer': 'pooler', 'name': 'smallm2', 'module': 'v10_sm2'}, {'m': 16, 'layer': 'qkv_gate', 'name': 'warps', 'module': 'v10_warps'}, {'m': 256, 'layer': 'o', 'name': 'm256tune', 'module': 'v10_tune', 'config': [256, 64, 5, 7, 1, 1, 1]}, {'m': 32, 'layer': 'qkv_gate', 'name': 'm32x', 'module': 'v10_m32x'}, {'m': 32, 'layer': 'pooler', 'name': 'm32x', 'module': 'v10_m32x'}]
SELECTED_ROUTES = [r for r in V9_SELECTED_ROUTES if (r["m"], r["layer"]) not in {(v["m"], v["layer"]) for v in V10_ROUTES}] + V10_ROUTES



def prepare_v10_tune(weights: Weights) -> dict:
    weight = weights.get('o')
    if weight is None or not weight.is_cuda or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (8192, 1024):
        return {}
    if torch.cuda.get_device_capability(weight.device) != (9, 0):
        return {}
    return {'ext': tt_v10_tune_build_rl([(256, 64, 5, 7, 1, 1, 1)]),
            'weight': tt_v10_tune_pack_weight(weight.contiguous(), 64)}


def run_v10_tune(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor | None:
    if layer != 'o' or n != 8192 or not state or x.ndim != 2 or tuple(x.shape) != (256, 1024):
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.data_ptr() % 16 or x.device != state['weight'].device:
        return None
    output = torch.empty((256, 8192), device=x.device, dtype=torch.bfloat16)
    rc = state['ext'].rl_launch(0, x.data_ptr(), state['weight'].data_ptr(), output.data_ptr(),
                                256, torch.cuda.current_stream(x.device).cuda_stream, 0)
    if rc:
        raise RuntimeError(f'm256tune O launch failed: {rc}')
    return output


def prepare(weights: Weights) -> dict:
    state = prepare_v9(weights)
    state['v10_sm2'] = tt_v10_sm2_prepare_smallm2(weights)
    state['v10_warps'] = tt_v10_warps_prepare_warps(weights)
    state['v10_tune'] = prepare_v10_tune(weights)
    state['v10_m32x'] = tt_v10_m32x_prepare_m32x(weights, include_unpooler=False)
    for route in V10_ROUTES:
        m, layer = route['m'], route['layer']
        weight = weights[layer]
        data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device),
                'config': {'m': m, 'n': weight.shape[0], 'k': weight.shape[1], 'layer': layer}}
        kernel_fn(data, state)
    torch.cuda.synchronize(weights['qkv_gate'].device)
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    output = None
    if layer == 'pooler' and m == 8:
        output = tt_v10_sm2_run_smallm2(x, layer, n, state['v10_sm2'])
    elif layer == 'qkv_gate' and m == 16:
        output = tt_v10_warps_run_warps(x, layer, n, state['v10_warps'])
    elif layer == 'o' and m == 256:
        output = run_v10_tune(x, layer, n, state['v10_tune'])
    elif layer in ('qkv_gate', 'pooler') and m == 32:
        output = tt_v10_m32x_run_m32x(x, layer, n, state['v10_m32x'])
    return candidate_v9(data, state) if output is None else output

