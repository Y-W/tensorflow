/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef CWISE_OPS_COMMON_H_
#define CWISE_OPS_COMMON_H_

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T, typename F, typename R = T>
struct base {
  // func defines operator() and its vectorized version packetOp().
  typedef F func;

  // If true, the functor's corresponding binary op will instantiate
  // specialized kernels to perform an optimized broadcast
  // operation. Each functor for which this is enabled increases the
  // code size, so by default this is disabled for binary functors and
  // is enabled on a per-op basis as needed.
  static const bool use_bcast_optimization = false;

  // operator() has the signature:
  //  out_type operator()(in_type in0, in_type in1 ...)
  typedef R out_type;
  typedef T in_type;

  // TensorFlow provides tensor-ized version of "func". Roughly
  // speaking, the tensorflow operation has the signature:
  //   tout_type op(tin_type in0)
  //   tout_type op(tin_type in0, tin_type in1)
  //   tout_type op(tin_type in0, in_type scalar)
  typedef typename TTypes<out_type>::Flat tout_type;
  typedef typename TTypes<in_type>::ConstFlat tin_type;
  typedef typename TTypes<in_type>::ConstScalar tscalar_type;

  // Whether the functor can error out.  Currently applies only to integer
  // div and mod.
  static const bool has_errors = false;
};

template <typename Device, typename Functor>
struct UnaryFunctor {
  // Computes on device "d": out[i] = Functor(in[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in);
};

template <typename D, typename Out, typename Rhs>
void Assign(const D& d, Out out, Rhs rhs) {
  out.device(d) = rhs;
}


// Partial specialization of UnaryFunctor<Device=CPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<CPUDevice, Functor> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    Assign(d, out, in.unaryExpr(typename Functor::func()));
  }
};

}  // end namespace functor

// Coefficient-wise unary operations:
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined in cwise_ops.h. E.g., functor::sqrt.
template <typename Device, typename Functor>
class UnaryOp : public OpKernel {
 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.
  // Tin may be different from Tout. E.g., abs: complex64 -> float

  explicit UnaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto in = DataTypeToEnum<Tin>::v();
    auto out = DataTypeToEnum<Tout>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({in}, {out}));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
    functor::UnaryFunctor<Device, Functor>()(
        ctx->eigen_device<Device>(), out->flat<Tout>(), inp.flat<Tin>());
  }
};

#define REGISTER(OP, D, N, F, T)                                             \
  REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_##D).TypeConstraint<T>("T"), \
                          OP<D##Device, F<T>>);

// Macros to register kernels for multiple types (T0, T1, etc.)  on
// device type "D" (CPU or GPU) for operation "N" (e.g., sqrt) using
// the functor "F" (e.g., functor:sqrt).

#define REGISTER2(OP, D, N, F, T0, T1) \
  REGISTER(OP, D, N, F, T0)            \
  REGISTER(OP, D, N, F, T1)
#define REGISTER3(OP, D, N, F, T0, T1, T2) \
  REGISTER2(OP, D, N, F, T0, T1)           \
  REGISTER(OP, D, N, F, T2)
#define REGISTER4(OP, D, N, F, T0, T1, T2, T3) \
  REGISTER2(OP, D, N, F, T0, T1)               \
  REGISTER2(OP, D, N, F, T2, T3)
#define REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4) \
  REGISTER3(OP, D, N, F, T0, T1, T2)               \
  REGISTER2(OP, D, N, F, T3, T4)
#define REGISTER6(OP, D, N, F, T0, T1, T2, T3, T4, T5) \
  REGISTER3(OP, D, N, F, T0, T1, T2)                   \
  REGISTER3(OP, D, N, F, T3, T4, T5)
#define REGISTER7(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6) \
  REGISTER4(OP, D, N, F, T0, T1, T2, T3)                   \
  REGISTER3(OP, D, N, F, T4, T5, T6)
#define REGISTER8(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7) \
  REGISTER4(OP, D, N, F, T0, T1, T2, T3)                       \
  REGISTER4(OP, D, N, F, T4, T5, T6, T7)
#define REGISTER9(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4)                       \
  REGISTER4(OP, D, N, F, T5, T6, T7, T8)

// Instead of adding REGISTER10, etc., shard the .cc files - see
// cwise_op_equal_to_*.cc for an example.

}  // end namespace tensorflow

#endif  // CWISE_OPS_COMMON_H_
