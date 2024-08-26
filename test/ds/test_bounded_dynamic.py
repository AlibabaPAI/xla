import os
import sys
import unittest
import torch, torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import test_utils

pd = torch._C._EnablePythonDispatcher()
dev = xm.xla_device()


def mark_dynamic(t, dims, bounds):
  torch_xla._XLAC._xla_mark_bounded_dynamic(t, dims, bounds)


class TestBoundedDynamicShapes(test_utils.XlaTestCase):

  def test_mark_dynamic(self):
    t1 = torch.randn([5, 2]).to(dev)
    # t1 has size [<=10, 2]
    mark_dynamic(t1, [0], [10])
    self.assertIn('<=10', torch_xla._XLAC._get_xla_tensors_text([t1]))
    if test_utils.is_disc_backend():
      t1_cpu = t1.cpu()
      self.assertEqual(t1_cpu.shape[0], 5)

  def test_sizeGe(self):
    met.clear_all()
    t1 = torch.randn([5, 2]).to(dev)
    # t1 has size [<=10, 2]
    mark_dynamic(t1, [0], [10])
    self.assertTrue(t1.shape[0] >= t1.shape[1])
    self.assertGreater(met.counter_value("xla::size_ge"), 0)
    self.assertIsNone(met.metric_data('CompileTime'))

  def test_sizeLt(self):
    met.clear_all()
    t1 = torch.randn([5, 2]).to(dev)
    # t1 has size [<=10, 2]
    mark_dynamic(t1, [0], [10])
    self.assertFalse(t1.shape[0] < t1.shape[1])
    self.assertGreater(met.counter_value("xla::size_lt"), 0)
    self.assertIsNone(met.metric_data('CompileTime'))

  def test_sizeNe(self):
    met.clear_all()
    t1 = torch.randn([5, 2]).to(dev)
    # t1 has size [<=10, 2]
    mark_dynamic(t1, [0], [10])
    self.assertTrue(t1.shape[0] != t1.shape[1])
    self.assertGreater(met.counter_value("xla::size_ne"), 0)
    self.assertIsNone(met.metric_data('CompileTime'))

  def test_sizeEq(self):
    met.clear_all()
    t1 = torch.randn([5, 2]).to(dev)
    # t1 has size [<=10, 2]
    mark_dynamic(t1, [0], [10])
    self.assertFalse(t1.shape[0] == 1)
    self.assertGreater(met.counter_value("xla::size_eq"), 0)
    self.assertIsNone(met.metric_data('CompileTime'))


if __name__ == '__main__':
  os.environ['USE_BOUND_FOR_SHAPE_COMPARE'] = os.getenv(
      'USE_BOUND_FOR_SHAPE_COMPARE', '1')
  test = unittest.main()
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
