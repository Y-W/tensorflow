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

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
#define TENSORFLOW_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_

#define EIGEN_USE_GPU

#include <complex>

#include "tensorflow/core/framework/tensor_types.h"
namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

// Partial specialization of UnaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    To32Bit(out).device(d) = To32Bit(in).unaryExpr(typename Functor::func());
  }
};

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for UnaryFunctor (e.g., functor::sqrt).
#define DEFINE_UNARY1(F, T) template struct UnaryFunctor<GPUDevice, F<T> >
#define DEFINE_UNARY2(F, T0, T1) \
  DEFINE_UNARY1(F, T0);          \
  DEFINE_UNARY1(F, T1)
#define DEFINE_UNARY3(F, T0, T1, T2) \
  DEFINE_UNARY2(F, T0, T1);          \
  DEFINE_UNARY1(F, T2)
#define DEFINE_UNARY4(F, T0, T1, T2, T3) \
  DEFINE_UNARY2(F, T0, T1);              \
  DEFINE_UNARY2(F, T2, T3)
#define DEFINE_UNARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_UNARY2(F, T0, T1);                  \
  DEFINE_UNARY3(F, T2, T3, T4)
#define DEFINE_UNARY6(F, T0, T1, T2, T3, T4, T5) \
  DEFINE_UNARY2(F, T0, T1);                      \
  DEFINE_UNARY4(F, T2, T3, T4, T5)

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for BinaryFunctor.
#define DEFINE_BINARY1(F, T)                         \
  template struct BinaryFunctor<GPUDevice, F<T>, 1>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 2>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 3>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 4>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 5>
#define DEFINE_BINARY2(F, T0, T1) \
  DEFINE_BINARY1(F, T0);          \
  DEFINE_BINARY1(F, T1)
#define DEFINE_BINARY3(F, T0, T1, T2) \
  DEFINE_BINARY2(F, T0, T1);          \
  DEFINE_BINARY1(F, T2)
#define DEFINE_BINARY4(F, T0, T1, T2, T3) \
  DEFINE_BINARY2(F, T0, T1);              \
  DEFINE_BINARY2(F, T2, T3)
#define DEFINE_BINARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_BINARY2(F, T0, T1);                  \
  DEFINE_BINARY3(F, T2, T3, T4)
#define DEFINE_BINARY6(F, T0, T1, T2, T3, T4, T5) \
  DEFINE_BINARY3(F, T0, T1, T2);                  \
  DEFINE_BINARY3(F, T3, T4, T5)
#define DEFINE_BINARY7(F, T0, T1, T2, T3, T4, T5, T6) \
  DEFINE_BINARY3(F, T0, T1, T2);                      \
  DEFINE_BINARY4(F, T3, T4, T5, T6)
#define DEFINE_BINARY8(F, T0, T1, T2, T3, T4, T5, T6, T7) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                      \
  DEFINE_BINARY4(F, T4, T5, T6, T7)
#define DEFINE_BINARY9(F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                          \
  DEFINE_BINARY5(F, T4, T5, T6, T7, T8)
#define DEFINE_BINARY10(F, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
  DEFINE_BINARY5(F, T0, T1, T2, T3, T4);                           \
  DEFINE_BINARY5(F, T5, T6, T7, T8, T9)
#define DEFINE_BINARY11(F, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
  DEFINE_BINARY5(F, T0, T1, T2, T3, T4);                                \
  DEFINE_BINARY6(F, T5, T6, T7, T8, T9, T10)

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
