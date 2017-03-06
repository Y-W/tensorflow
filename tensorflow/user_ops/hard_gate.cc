/***************************************
 Hard Gating Function (i.e. binary step function)
 Yijie Wang (wyijie93@gmail.com)
 ***************************************/

#include "tensorflow/core/framework/common_shape_fns.h"

#include "hard_gate.h"

namespace tensorflow {

REGISTER_OP("HardGate")
  .Input("continuous_input: T")
  .Output("binary_output: bool")
  .Attr("T: {half, float, double, int32, int64}")
  .SetShapeFn(::tensorflow::shape_inference::UnchangedShape)
  .Doc(R"doc(
Compute the binary step function.

`y = 0` if `x <= 0`; 1 if `x > 0`.
)doc");

REGISTER4(UnaryOp, CPU, "HardGate", functor::hard_gate, float, double, int32, int64);

}  // namespace tensorflow


