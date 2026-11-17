/* convert_<from>_<to>: float <-> float. One work item per `group` elements
 * (lcm of both block sizes), so packed outputs never share a byte between
 * work items. */
#include "nb_codec.h"

__kernel void nb_convert(const __global uchar *src, __global uchar *dst, uint len, uint sat,
                         uint from, uint to, uint group) {
    uint first = NB_GID() * group;
    if (first >= len) return;
    nb_zero_packed(dst, to, first, group);
    for (uint i = first; i < first + group; i++) nb_store_f(dst, to, i, nb_load_f(src, from, i), sat);
}
