/* Target layer: the only place that differs between AMDGPU, NVPTX, SPIR-V
 * and the C builds (NB_HOST): the host test driver and the freestanding
 * Tenstorrent RISC-V runtime (kernels/tt).
 *
 * Every build uses -ffp-contract=off: results must match the Rust oracle,
 * which never fuses a*b+c unless it calls fma explicitly. */
#ifndef NB_TARGET_H
#define NB_TARGET_H

#if defined(NB_HOST)
#include <stdint.h>
typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;
#define __kernel
#define __global
#define NB_INLINE static inline
extern uint nb_host_gid;
#define NB_GID() nb_host_gid
NB_INLINE uint nb_f2u(float x) { uint u; __builtin_memcpy(&u, &x, 4); return u; }
NB_INLINE float nb_u2f(uint u) { float x; __builtin_memcpy(&x, &u, 4); return x; }
#pragma STDC FP_CONTRACT OFF
#else
#pragma OPENCL FP_CONTRACT OFF
#define NB_INLINE static inline
#define nb_f2u(x) as_uint(x)
#define nb_u2f(x) as_float(x)
#if defined(__AMDGCN__)
#define NB_GID() ((uint)(__builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x() + __builtin_amdgcn_workitem_id_x()))
#elif defined(__NVPTX__)
#define NB_GID() ((uint)(__nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x()))
#else
#define NB_GID() ((uint)get_global_id(0))
#endif
#endif

/* No device library is linked (-nogpulib): only compiler builtins. */
#define nb_fma(a, b, c) __builtin_fmaf((a), (b), (c))
#define nb_fabs(x) nb_u2f(nb_f2u(x) & 0x7FFFFFFFu)
#define NB_NAN nb_u2f(0x7FC00000u)

#endif /* NB_TARGET_H */
