/* One translation unit with every Emulated kernel; build-kernels.sh compiles
 * this file once per target. */
#include "convert.cl"
#include "gemm.cl"
#include "image.cl"
#include "int.cl"
#include "quantize.cl"
#include "reduce.cl"
