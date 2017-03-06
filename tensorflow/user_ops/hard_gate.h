/***************************************
 Hard Gating Function (i.e. binary step function)
 Yijie Wang (wyijie93@gmail.com)
 ***************************************/

#include "tensorflow/core/framework/op.h"
#include "cwise_ops_common.h"

#ifndef HARD_GATE_H_
#define HARD_GATE_H_

namespace Eigen {
namespace internal {

template<typename T>
struct scalar_hard_gate_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_hard_gate_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const bool
  operator()(const T& a) const {
    return (a > 0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& a) const {
    return (a > 0);
  }
};
template <typename T>
struct functor_traits<scalar_hard_gate_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = packet_traits<T>::HasSign,
  };
};


}  // end namespace internal
}  // end namespace Eigen

namespace tensorflow {

namespace functor {
    template <typename T>
    struct hard_gate : base<T, Eigen::internal::scalar_hard_gate_op<T>, bool> {};
}  // namespace functor

}  // namespace tensorflow

#endif  // HARD_GATE_H_
