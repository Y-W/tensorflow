import os
import tensorflow as tf

_hard_gate_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'hard_gate.so'))
hard_gate = _hard_gate_module.hard_gate

from tensorflow.python.framework import ops

# Straight-through Estimator
@ops.RegisterGradient("HardGate")
def _hard_gate_grad(op, grad):
    return [grad]

