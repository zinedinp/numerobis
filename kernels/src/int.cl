/* pack_<to> / unpack_<from>: I8/U8 <-> packed I4/U4/I2/I1, saturating.
 * One work item per `group` elements (whole bytes of the packed side). */
#include "nb_codec.h"

__kernel void nb_int_convert(const __global uchar *src, __global uchar *dst, uint len, uint from, uint to,
                             uint group) {
    uint first = NB_GID() * group;
    if (first >= len) return;
    nb_zero_packed(dst, to, first, group);
    for (uint i = first; i < first + group; i++) nb_store_i(dst, to, i, nb_load_i(src, from, i));
}
