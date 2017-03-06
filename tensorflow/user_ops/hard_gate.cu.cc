/***************************************
 Hard Gating Function (i.e. binary step function)
 Yijie Wang (wyijie93@gmail.com)
 ***************************************/

#include "cwise_ops_gpu_common.cu.h"

#include "hard_gate.h"

namespace tensorflow {

#if GOOGLE_CUDA

/******************
namespace functor {
DEFINE_UNARY1(sign, float);
}  // namespace functor
******************/

REGISTER3(UnaryOp, GPU, "HardGate", functor::hard_gate, float, double, int64);

#endif

}  // namespace tensorflow


