import os
import re
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import utils
from torch.export import Dim, export
from torch_xla.stablehlo import exported_program_to_stablehlo

try:
  from torch_xla.tf_saved_model_integration import \
      save_torch_module_as_tf_saved_model
except ImportError:
  print("tf is not installed. The tf.saved_model tests will be skipped.")
from utils import (compare_exported_program_and_saved_model_result,
                   has_tf_package, wrap_func_as_nn_module)

device = xm.xla_device()
os.environ['EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM'] = '1'


class UnboundedDynamismExportTest(unittest.TestCase):

  def _test_export_dynamism_wrapper(self, f, args, dynamic_shapes):
    m = wrap_func_as_nn_module(f)
    ep = export(m, args=args, constraints=constraints)
    return ep

  def test_add(self):
    args = (torch.rand((10, 197, 768)), torch.rand((10, 197, 768)))
    dynamic_shapes = ([{0: Dim("dim")}, {0: Dim("dim")}],)
    m = wrap_func_as_nn_module(torch.ops.aten.add.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<\?x197x768xf32>.*tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_add_scalar(self):
    args = (torch.rand((10, 197, 768)), 0.345)
    dynamic_shapes = ([{0: Dim("dim")}, None],)
    m = wrap_func_as_nn_module(torch.ops.aten.add.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<f32>.*tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_addmm(self):
    args = (torch.rand((5)), torch.rand((10, 5)), torch.rand((5, 5)))
    dynamic_shapes = ([None, {0: Dim("dim")}, None],)
    m = wrap_func_as_nn_module(torch.ops.aten.addmm.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x5xf32>.*->.*tensor<\?x5xf32>', shlo_text)
        is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        # Hit stablehlo.dot shape refinement error when inferencing saved_model in TF.
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_bmm(self):
    args = (
        torch.rand((24, 197, 64)),
        torch.rand((24, 64, 197)),
    )
    dynamic_shapes = ([{0: Dim("dim")}, {0: Dim("dim")}],)
    m = wrap_func_as_nn_module(torch.ops.aten.bmm.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<\?x64x197xf32>.*%arg.: tensor<\?x197x64xf32>.*->.*tensor<\?x197x197xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_cat(self):
    args = (torch.rand((10, 1, 768)), torch.rand((10, 196, 768)))
    dynamic_shapes = ([{0: Dim("dim")}, {0: Dim("dim")}],)
    m = wrap_func_as_nn_module(
        lambda x, y: torch.ops.aten.cat.default([x, y], 1))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<\?x196x768xf32>.*%arg.: tensor<\?x1x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_conv(self):
    args = (
        torch.rand((10, 3, 224, 224)),
        torch.rand((5, 3, 16, 16)),
        torch.rand((5)),
    )
    dynamic_shapes = ([{0: Dim("dim")}, None, None],)
    m = wrap_func_as_nn_module(
        lambda x, y, z: torch.ops.aten.convolution.default(
            x,
            y,
            z,
            [16, 16],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        ))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x3x224x224xf32>.*->.*tensor<\?x5x14x14xf32>',
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_div(self):
    args = (torch.rand((10, 12, 197)), 8.0)
    dynamic_shapes = ([{0: Dim("dim")}, None],)
    m = wrap_func_as_nn_module(torch.ops.aten.div.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x12x197xf32>.*->.*tensor<\?x12x197xf32>',
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  @unittest.skip("xla::Erf doesn't support unbounded dynamic input.")
  def test_gelu(self):
    args = (torch.rand((3, 5)),)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.gelu, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    # shlo_text = shlo_module.get_stablehlo_text()
    # self.assertTrue(
    #     "(%arg0: tensor<?x2xi64>, %arg1: tensor<?x2xi64>) -> tensor<?x2xi64>" in
    #     shlo_text)

  @unittest.skip("Unbounded Dynamism not supported yet.")
  def test_native_layer_norm(self):
    args = (
        torch.rand((10, 197, 768)),
        [768],
        torch.rand((768)),
        torch.rand((768)),
        1e-12,
    )
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(
        torch.ops.aten.native_layer_norm.default, args, constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>",
                  shlo_text) is not None)

  def test_permute(self):
    args = (torch.rand((10, 197, 12, 64)),)
    dynamic_shapes = ([{0: Dim("dim")}],)
    m = wrap_func_as_nn_module(
        lambda x: torch.ops.aten.permute.default(x, [0, 2, 1, 3]))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x197x12x64xf32>.*->.*tensor<\?x12x197x64xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  @unittest.skip("Unbounded Dynamism not supported yet.")
  def test_select(self):
    args = (torch.rand((10, 197, 768)), 1, 0)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.select.int, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x768xf32>",
                  shlo_text) is not None)

  def test_slice(self):
    args = (torch.rand((10, 3, 224, 224)), 0, 0, 9223372036854775807)
    dynamic_shapes = ([{0: Dim("dim")}, None, None, None],)
    m = wrap_func_as_nn_module(torch.ops.aten.slice.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x3x224x224xf32>.*->.*tensor<\?x3x224x224xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_softmax(self):
    args = (torch.rand((10, 12, 197, 197)), -1, False)
    dynamic_shapes = ([{0: Dim("dim")}, None, None],)
    m = wrap_func_as_nn_module(torch.ops.aten._softmax.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x12x197x197xf32>.*->.*tensor<\?x12x197x197xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
