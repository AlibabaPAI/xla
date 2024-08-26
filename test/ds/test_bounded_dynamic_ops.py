import os
import sys
import unittest
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import test_utils

pd = torch._C._EnablePythonDispatcher()
dev = xm.xla_device()


def mark_dynamic(t, dims, bounds):
  torch_xla._XLAC._xla_mark_bounded_dynamic(t, dims, bounds)


def diff_output(testcase,
                output1,
                output2,
                atol=1e-3,
                rtol=1e-5,
                equal_nan=True):
  if isinstance(output1, torch.Tensor):
    testcase.assertIsInstance(output2, torch.Tensor)
    output2_cpu = output2.detach().cpu()
    if output2_cpu.dtype != output1.dtype:
      output2_cpu = output2_cpu.to(output1.dtype)
    testcase.assertEqual(output1.shape, output2.shape)
    testcase.assertTrue(
        torch.allclose(
            output1, output2_cpu, atol=atol, rtol=rtol, equal_nan=equal_nan))
  elif isinstance(output1, (tuple, list)):
    testcase.assertIsInstance(output2, (tuple, list))
    testcase.assertEqual(len(output1), len(output2))
    for o1, o2 in zip(output1, output2):
      diff_output(testcase, o1, o2, rtol, atol)
  else:
    testcase.assertEqual(output1, output2)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      device: torch.device,
                      past_key_values_length: int = 0):
  """
  Make causal mask used for bi-directional self-attention.
  """
  bsz, tgt_len = input_ids_shape
  mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
  mask_cond = torch.arange(mask.size(-1), device=device)
  mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
  mask = mask.to(dtype)

  if past_key_values_length > 0:
    mask = torch.cat([
        torch.zeros(
            tgt_len, past_key_values_length, dtype=dtype, device=device), mask
    ],
                     dim=-1)
  return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                       tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len=None):
  """
  Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
  """
  bsz, src_len = mask.size()
  tgt_len = tgt_len if tgt_len is not None else src_len

  expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                src_len).to(dtype)

  inverted_mask = 1.0 - expanded_mask

  return inverted_mask.masked_fill(
      inverted_mask.to(torch.bool),
      torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds,
                                    past_key_values_length):
  # create causal mask
  # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
  combined_attention_mask = None
  if input_shape[-1] > 1:
    combined_attention_mask = _make_causal_mask(
        input_shape,
        inputs_embeds.dtype,
        device=inputs_embeds.device,
        past_key_values_length=past_key_values_length,
    )

  if attention_mask is not None:
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_attn_mask = _expand_mask(
        attention_mask, inputs_embeds.dtype,
        tgt_len=input_shape[-1]).to(inputs_embeds.device)
    combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else
        expanded_attn_mask + combined_attention_mask)

  return combined_attention_mask


class TestBoundedDynamicOps(test_utils.XlaTestCase):

  def diff_output(self,
                  torch_out,
                  xla_out,
                  atol=1e-3,
                  rtol=1e-5,
                  equal_nan=True):
    if isinstance(torch_out, torch.Tensor):
      self.assertIsInstance(xla_out, torch.Tensor)
      torch_out = torch_out.detach().cpu()
      xla_out = xla_out.detach().cpu()
      if xla_out.dtype != torch_out.dtype:
        xla_out = xla_out.to(torch_out.dtype)
      self.assertEqual(torch_out.shape, xla_out.shape)
      self.assertTrue(
          torch.allclose(
              torch_out, xla_out, atol=atol, rtol=rtol, equal_nan=equal_nan))
    elif isinstance(torch_out, (tuple, list)):
      self.assertIsInstance(xla_out, (tuple, list))
      self.assertEqual(len(torch_out), len(xla_out))
      for o1, o2 in zip(torch_out, xla_out):
        self.diff_output(o1, o2, rtol, atol)
    else:
      self.assertEqual(torch_out, xla_out)

  def test_add(self):
    t1 = torch.randn([5, 2])
    t2 = torch.randn([5, 2])
    torch_out = t1 + t2

    t1 = t1.to(dev)
    t2 = t2.to(dev)
    mark_dynamic(t1, [0], [10])
    mark_dynamic(t2, [0], [10])
    xla_out = t1 + t2
    self.diff_output(torch_out, xla_out)

  def test_add_broadcast(self):
    t1 = torch.randn([5, 2])
    t2 = torch.randn([2])
    torch_out = t1 + t2

    t1 = t1.to(dev)
    t2 = t2.to(dev)
    mark_dynamic(t1, [0], [10])
    xla_out = t1 + t2
    self.diff_output(torch_out, xla_out)

  def test_add_scalar(self):
    t1 = torch.randn([5, 2])
    t2 = 1.0
    torch_out = t1 + t2

    t1 = t1.to(dev)
    mark_dynamic(t1, [0], [10])
    xla_out = t1 + t2
    self.diff_output(torch_out, xla_out)

  def test_reshape(self):
    x = torch.randn(4, 101, 100)
    y = torch.randn(4 * 101 * 100)
    torch_out = y.reshape(x.shape[0], x.shape[1], -1)

    x = x.to(dev)
    y = y.to(dev)
    mark_dynamic(x, [0, 1], [10, 200])
    mark_dynamic(y, [0], [10 * 200 * 100])
    xla_out = y.reshape(x.shape[0], x.shape[1], -1)
    self.diff_output(torch_out, xla_out)

  def test_flatten(self):
    x = torch.randn(4, 101, 100)
    torch_out = x.flatten(0, 1)

    x = x.to(dev)
    mark_dynamic(x, [0], [10])
    xla_out = x.flatten(0, 1)
    self.diff_output(torch_out, xla_out)

  def test_arange(self):
    x = torch.randn(4, 101, 100)
    torch_out = torch.arange(
        0, (x.shape[0] + 1) * x.shape[1],
        step=x.shape[1],
        dtype=torch.int32,
        device=x.device)

    x = x.to(dev)
    mark_dynamic(x, [1], [200])
    xla_out = torch.arange(
        0, (x.shape[0] + 1) * x.shape[1],
        step=x.shape[1],
        dtype=torch.int32,
        device=x.device)
    self.diff_output(torch_out, xla_out)

  def test_slice_with_backward(self):
    x = torch.randn(4, 101, 100)
    y = torch.randn(4, 201, 100)
    x.requires_grad = True
    y.requires_grad = True
    torch_out = y[0:10, 10:x.shape[1], ...]
    torch.autograd.backward(torch_out, torch.zeros_like(torch_out))
    torch_grad = y.grad

    x = x.detach().to(dev)
    y = y.detach().to(dev)
    x.requires_grad = True
    y.requires_grad = True
    mark_dynamic(x, [1], [200])
    xla_out = y[0:10, 10:x.shape[1], ...]
    torch.autograd.backward(xla_out, torch.zeros_like(xla_out))
    xla_grad = y.grad

    self.diff_output(torch_out, xla_out)
    self.diff_output(torch_grad, xla_grad)

  def test_attn_mask(self):
    inputs_embeds = torch.randn(4, 101)
    attention_mask = torch.ones((4, 101),
                                dtype=torch.bool).to(inputs_embeds.device)
    torch_out = _prepare_decoder_attention_mask(
        attention_mask, (inputs_embeds.shape[0], inputs_embeds.shape[1]),
        inputs_embeds, 0)

    inputs_embeds = inputs_embeds.to(dev)
    attention_mask = attention_mask.to(dev)
    mark_dynamic(inputs_embeds, [1], [200])
    mark_dynamic(attention_mask, [1], [200])
    xla_out = _prepare_decoder_attention_mask(
        attention_mask, (inputs_embeds.shape[0], inputs_embeds.shape[1]),
        inputs_embeds, 0)

    self.diff_output(torch_out, xla_out)

  def test_matmul_0(self):
    t1 = torch.randn([5, 2]).to(torch.bfloat16)
    t2 = torch.randn([2, 3]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(dev)
    t2 = t2.to(dev)
    mark_dynamic(t1, [0], [10])
    xla_out = t1 @ t2

    self.assertIn('<=10,3', torch_xla._XLAC._get_xla_tensors_text([xla_out]))
    self.diff_output(torch_out, xla_out)

  def test_matmul_1(self):
    t1 = torch.randn([5, 2]).to(torch.bfloat16)
    t2 = torch.randn([2]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(dev)
    t2 = t2.to(dev)
    mark_dynamic(t1, [0], [10])
    xla_out = t1 @ t2

    self.assertIn('<=10', torch_xla._XLAC._get_xla_tensors_text([xla_out]))
    self.diff_output(torch_out, xla_out)

  def test_matmul_2(self):
    t1 = torch.randn([10, 5, 2]).to(torch.bfloat16)
    t2 = torch.randn([2]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(dev)
    t2 = t2.to(dev)
    mark_dynamic(t1, [0, 1], [20, 10])
    xla_out = t1 @ t2
    self.assertIn('<=20,<=10', torch_xla._XLAC._get_xla_tensors_text([xla_out]))
    self.diff_output(torch_out.cpu(), xla_out)

  def test_matmul_3(self):
    t1 = torch.randn([10, 3, 4]).to(torch.bfloat16)
    t2 = torch.randn([10, 4, 5]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(dev)
    t2 = t2.to(dev)
    mark_dynamic(t1, [0, 1], [20, 10])
    mark_dynamic(t2, [0], [20])
    xla_out = t1 @ t2
    self.assertIn('<=20,<=10,5',
                  torch_xla._XLAC._get_xla_tensors_text([xla_out]))
    self.diff_output(torch_out, xla_out)

  def test_matmul_4(self):
    t1 = torch.randn([10, 3, 4]).to(torch.bfloat16)
    t2 = torch.randn([4, 5]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(dev)
    t2 = t2.to(dev)
    mark_dynamic(t1, [0, 1], [20, 10])
    xla_out = t1 @ t2
    self.assertIn('<=20,<=10,5',
                  torch_xla._XLAC._get_xla_tensors_text([xla_out]))
    self.diff_output(torch_out, xla_out)

  def test_triu(self):
    t = torch.randn(4, 4)
    torch_out = torch.triu(t, diagonal=1)

    t = t.to(dev)
    mark_dynamic(t, [0, 1], [10, 10])
    xla_out = torch.triu(t, diagonal=1)

    self.assertIn('<=10,<=10', torch_xla._XLAC._get_xla_tensors_text([xla_out]))
    self.diff_output(torch_out, xla_out)

  def test_nll_loss_with_backward(self):
    logits = torch.randn(20, 30)
    target = torch.randint(0, 30, (20,), dtype=torch.long)
    logits.requires_grad = True
    torch_out = F.nll_loss(logits, target)
    torch_out.backward()
    torch_grad = logits.grad

    logits = logits.detach().to(dev)
    logits.requires_grad = True
    target = target.to(dev)
    mark_dynamic(logits, [0], [50])
    mark_dynamic(target, [0], [50])
    xla_out = F.nll_loss(logits, target)
    xla_out.backward()
    xla_grad = logits.grad

    self.diff_output(torch_out, xla_out)
    self.diff_output(torch_grad, xla_grad)


if __name__ == '__main__':
  assert test_utils.is_disc_backend()
  os.environ['USE_BOUND_FOR_SHAPE_COMPARE'] = os.getenv(
      'USE_BOUND_FOR_SHAPE_COMPARE', '1')
  test = unittest.main()
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
