import tensorflow as tf
from hard_gate import hard_gate

def make_cases():
  a = tf.constant([0.0, -2.0, 3.0, 4.0, -5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([-1, -2, 0, -4, 6], shape=[5,], name='b')
  out_a = hard_gate(a)
  out_b = hard_gate(b)
  grad_a = tf.gradients(tf.sum(out_a), a)
  return out_a, out_b, grad_a

def main():
  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device('/cpu:0'):
      oa, ob, ga = make_cases()
      print sess.run(oa)
      print sess.run(ob)
      print sess.run(ga)
    with tf.device('/gpu:0'):
      oa, ob, ga = make_cases()
      print sess.run(oa)
      print sess.run(ob)
      print sess.run(ga)

if __name__ == "__main__":
  main()

