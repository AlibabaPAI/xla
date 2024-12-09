import os
import sys
import unittest
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input
import flash_attn_2_cuda as flash_attn_cuda

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import test_utils
from test_flash_attention_varlen_forward import _unpad_input as _fwd_unpad_input
from test_flash_attention_varlen_backward import _unpad_input as _bwd_unpad_input

PD = torch._C._EnablePythonDispatcher()
XLA_DEVICE = xm.xla_device()


def _mark_dynamic(t, dims, bounds):
  torch_xla._XLAC._xla_mark_bounded_dynamic(t, dims, bounds)


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

  def _diff_output(self,
                   torch_out,
                   xla_out,
                   atol=1e-3,
                   rtol=1e-5,
                   equal_nan=True):
    if isinstance(torch_out, torch.Tensor):
      self.assertIsInstance(xla_out, torch.Tensor)
      torch_out = torch_out.detach().cpu()
      xla_out = xla_out.detach().cpu()
      self.assertEqual(xla_out.dtype, torch_out.dtype)
      self.assertEqual(torch_out.shape, xla_out.shape)
      self.assertTrue(
          torch.allclose(
              torch_out, xla_out, atol=atol, rtol=rtol, equal_nan=equal_nan))
    elif isinstance(torch_out, (tuple, list)):
      self.assertIsInstance(xla_out, (tuple, list))
      self.assertEqual(len(torch_out), len(xla_out))
      for o1, o2 in zip(torch_out, xla_out):
        self._diff_output(o1, o2, rtol, atol)
    else:
      self.assertEqual(torch_out, xla_out)

  def test_add(self):
    t1 = torch.randn([5, 2])
    t2 = torch.randn([5, 2])
    torch_out = t1 + t2

    t1 = t1.to(XLA_DEVICE)
    t2 = t2.to(XLA_DEVICE)
    _mark_dynamic(t1, [0], [10])
    _mark_dynamic(t2, [0], [10])
    xla_out = t1 + t2
    self._diff_output(torch_out, xla_out)

  def test_add_broadcast(self):
    t1 = torch.randn([5, 2])
    t2 = torch.randn([2])
    torch_out = t1 + t2

    t1 = t1.to(XLA_DEVICE)
    t2 = t2.to(XLA_DEVICE)
    _mark_dynamic(t1, [0], [10])
    xla_out = t1 + t2
    self._diff_output(torch_out, xla_out)

  def test_add_scalar(self):
    t1 = torch.randn([5, 2])
    t2 = 1.0
    torch_out = t1 + t2

    t1 = t1.to(XLA_DEVICE)
    _mark_dynamic(t1, [0], [10])
    xla_out = t1 + t2
    self._diff_output(torch_out, xla_out)

  def test_reshape(self):
    x = torch.randn(4, 101, 100)
    y = torch.randn(4 * 101 * 100)
    torch_out = y.reshape(x.shape[0], x.shape[1], -1)

    x = x.to(XLA_DEVICE)
    y = y.to(XLA_DEVICE)
    _mark_dynamic(x, [0, 1], [10, 200])
    _mark_dynamic(y, [0], [10 * 200 * 100])
    xla_out = y.reshape(x.shape[0], x.shape[1], -1)
    self._diff_output(torch_out, xla_out)

  def test_flatten(self):
    x = torch.randn(4, 101, 100)
    torch_out = x.flatten(0, 1)

    x = x.to(XLA_DEVICE)
    _mark_dynamic(x, [0], [10])
    xla_out = x.flatten(0, 1)
    self._diff_output(torch_out, xla_out)

  def test_arange(self):
    x = torch.randn(4, 101, 100)
    torch_out = torch.arange(
        0, (x.shape[0] + 1) * x.shape[1],
        step=x.shape[1],
        dtype=torch.int32,
        device=x.device)

    x = x.to(XLA_DEVICE)
    _mark_dynamic(x, [1], [200])
    xla_out = torch.arange(
        0, (x.shape[0] + 1) * x.shape[1],
        step=x.shape[1],
        dtype=torch.int32,
        device=x.device)
    self._diff_output(torch_out, xla_out)

  def test_slice_with_backward(self):
    x = torch.randn(4, 101, 100)
    y = torch.randn(4, 201, 100)
    x.requires_grad = True
    y.requires_grad = True
    torch_out = y[0:10, 10:x.shape[1], ...]
    torch.autograd.backward(torch_out, torch.zeros_like(torch_out))
    torch_grad = y.grad

    x = x.detach().to(XLA_DEVICE)
    y = y.detach().to(XLA_DEVICE)
    x.requires_grad = True
    y.requires_grad = True
    _mark_dynamic(x, [1], [200])
    xla_out = y[0:10, 10:x.shape[1], ...]
    torch.autograd.backward(xla_out, torch.zeros_like(xla_out))
    xla_grad = y.grad

    self._diff_output(torch_out, xla_out)
    self._diff_output(torch_grad, xla_grad)

  def test_attn_mask(self):
    inputs_embeds = torch.randn(4, 101)
    attention_mask = torch.ones((4, 101),
                                dtype=torch.bool).to(inputs_embeds.device)
    torch_out = _prepare_decoder_attention_mask(
        attention_mask, (inputs_embeds.shape[0], inputs_embeds.shape[1]),
        inputs_embeds, 0)

    inputs_embeds = inputs_embeds.to(XLA_DEVICE)
    attention_mask = attention_mask.to(XLA_DEVICE)
    _mark_dynamic(inputs_embeds, [1], [200])
    _mark_dynamic(attention_mask, [1], [200])
    xla_out = _prepare_decoder_attention_mask(
        attention_mask, (inputs_embeds.shape[0], inputs_embeds.shape[1]),
        inputs_embeds, 0)

    self._diff_output(torch_out, xla_out)

  def test_matmul_0(self):
    t1 = torch.randn([5, 2]).to(torch.bfloat16)
    t2 = torch.randn([2, 3]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(XLA_DEVICE)
    t2 = t2.to(XLA_DEVICE)
    _mark_dynamic(t1, [0], [10])
    xla_out = t1 @ t2

    self.assertIn('<=10, 3', str(xla_out.shape))
    self._diff_output(torch_out, xla_out)

  def test_matmul_1(self):
    t1 = torch.randn([5, 2]).to(torch.bfloat16)
    t2 = torch.randn([2]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(XLA_DEVICE)
    t2 = t2.to(XLA_DEVICE)
    _mark_dynamic(t1, [0], [10])
    xla_out = t1 @ t2

    self.assertIn('<=10', str(xla_out.shape))
    self._diff_output(torch_out, xla_out)

  def test_matmul_2(self):
    t1 = torch.randn([10, 5, 2]).to(torch.bfloat16)
    t2 = torch.randn([2]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(XLA_DEVICE)
    t2 = t2.to(XLA_DEVICE)
    _mark_dynamic(t1, [0, 1], [20, 10])
    xla_out = t1 @ t2
    self.assertIn('<=20, <=10', str(xla_out.shape))
    self._diff_output(torch_out.cpu(), xla_out)

  def test_matmul_3(self):
    t1 = torch.randn([10, 3, 4]).to(torch.bfloat16)
    t2 = torch.randn([10, 4, 5]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(XLA_DEVICE)
    t2 = t2.to(XLA_DEVICE)
    _mark_dynamic(t1, [0, 1], [20, 10])
    _mark_dynamic(t2, [0], [20])
    xla_out = t1 @ t2
    self.assertIn('<=20, <=10, 5', str(xla_out.shape))
    self._diff_output(torch_out, xla_out)

  def test_matmul_4(self):
    t1 = torch.randn([10, 3, 4]).to(torch.bfloat16)
    t2 = torch.randn([4, 5]).to(torch.bfloat16)
    torch_out = t1.to("cuda") @ t2.to("cuda")

    t1 = t1.to(XLA_DEVICE)
    t2 = t2.to(XLA_DEVICE)
    _mark_dynamic(t1, [0, 1], [20, 10])
    xla_out = t1 @ t2
    self.assertIn('<=20, <=10, 5', str(xla_out.shape))
    self._diff_output(torch_out, xla_out)

  def test_triu(self):
    t = torch.randn(4, 4)
    torch_out = torch.triu(t, diagonal=1)

    t = t.to(XLA_DEVICE)
    _mark_dynamic(t, [0, 1], [10, 10])
    xla_out = torch.triu(t, diagonal=1)

    self.assertIn('<=10, <=10', str(xla_out.shape))
    self._diff_output(torch_out, xla_out)

  def test_nll_loss_with_backward(self):
    logits = torch.randn(20, 30)
    target = torch.randint(0, 30, (20,), dtype=torch.long)
    logits.requires_grad = True
    torch_out = F.nll_loss(logits, target)
    torch_out.backward()
    torch_grad = logits.grad

    logits = logits.detach().to(XLA_DEVICE)
    logits.requires_grad = True
    target = target.to(XLA_DEVICE)
    _mark_dynamic(logits, [0], [50])
    _mark_dynamic(target, [0], [50])
    xla_out = F.nll_loss(logits, target)
    xla_out.backward()
    xla_grad = logits.grad
    self.assertIn('<=50, 30', str(xla_grad.shape))

    self._diff_output(torch_out, xla_out)
    self._diff_output(torch_grad, xla_grad)

  def test_flash_attn_fwd(self):
    d = 32
    batch_size = 4
    nheads = 9
    nheads_k = nheads
    window_size = (-1, -1)

    seqlen_q = 2048
    seqlen_k = 2048

    device = "cuda"
    dtype = torch.bfloat16

    alibi = False
    dropout_p = 0.0
    causal = True
    softmax_scale = 0.25
    deterministic = False

    q = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=False)
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=False)
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=False)

    attention_mask = torch.zeros(
        batch_size, seqlen_k, dtype=torch.int32).to(device)

    k_lengths = torch.randint(low=2, high=seqlen_k, size=(batch_size,))

    for i in range(batch_size):
      k_len = k_lengths[i].item()
      attention_mask[i, :k_len] = 1
      q[i, k_len:, :, :] = 0
      k[i, k_len:, :, :] = 0
      v[i, k_len:, :, :] = 0
    q_cuda, k_cuda, v_cuda, indices_q, cu_seq_lens, max_seq_lens = _fwd_unpad_input(
        q, k, v, attention_mask, seqlen_q, nheads)
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

    if alibi:
      alibi_slopes = torch.rand(
          batch_size, nheads, device=device, dtype=torch.float32) * 0.3
    else:
      alibi_slopes = None

    out_fa, softmax_lse, _ = flash_attn_varlen_func(
        q_cuda.contiguous(),
        k_cuda.contiguous(),
        v_cuda.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    out_fa = pad_input(out_fa, indices_q, batch_size, seqlen_q)

    q = q.cpu().detach()
    k = k.cpu().detach()
    v = v.cpu().detach()
    out_fa = out_fa.cpu().detach()
    softmax_lse = softmax_lse.cpu().detach()
    cu_seqlens_q = cu_seqlens_q.cpu().detach()
    cu_seqlens_k = cu_seqlens_k.cpu().detach()
    if alibi:
      alibi_slopes = alibi_slopes.cpu()
    torch.cuda.synchronize()

    device = XLA_DEVICE
    torch.random.manual_seed(0)
    q_xla = q.to(device)
    k_xla = k.to(device)
    v_xla = v.to(device)
    attention_mask_xla = attention_mask.to(device)

    _mark_dynamic(q_xla, [1], [seqlen_q + 100])
    _mark_dynamic(k_xla, [1], [seqlen_k + 100])
    _mark_dynamic(v_xla, [1], [seqlen_k + 100])
    _mark_dynamic(attention_mask_xla, [1], [seqlen_k + 100])

    q_xla.requires_grad = False
    k_xla.requires_grad = False
    v_xla.requires_grad = False
    if alibi:
      alibi_slopes = alibi_slopes.cpu().to(device)
    softmax_lse_xla, out_xla, _, cu_seqlen_q_xla, cu_seqlen_k_xla = torch_xla._XLAC._flash_attention_forward(
        q_xla.contiguous(), k_xla.contiguous(), v_xla.contiguous(),
        attention_mask_xla.contiguous(), alibi_slopes, dropout_p, softmax_scale,
        False, causal, window_size[0], window_size[1], True, None)

    self.assertIn(f'{batch_size}, <={seqlen_q+100}, {nheads}, {d}',
                  str(out_xla.shape))
    self.assertIn(f'{batch_size}, {nheads}, <={seqlen_q+100}',
                  str(softmax_lse_xla.shape))

    xm.mark_step(wait=True)
    q_xla = q_xla.cpu().detach()
    k_xla = k_xla.cpu().detach()
    v_xla = v_xla.cpu().detach()
    out_xla = out_xla.cpu().detach()
    cu_seqlen_q_xla = cu_seqlen_q_xla.cpu().detach()
    cu_seqlen_k_xla = cu_seqlen_k_xla.cpu().detach()
    softmax_lse_xla = softmax_lse_xla.cpu().detach()
    attention_mask_xla = attention_mask_xla.cpu().detach()

    self.assertTrue(
        torch.allclose(q_xla, q, rtol=1e-3, atol=1e-3, equal_nan=True))
    self.assertTrue(
        torch.allclose(k_xla, k, rtol=1e-3, atol=1e-3, equal_nan=True))
    self.assertTrue(
        torch.allclose(v_xla, v, rtol=1e-3, atol=1e-3, equal_nan=True))
    self.assertTrue(
        torch.allclose(out_xla, out_fa, rtol=1e-2, atol=1e-2, equal_nan=True))
    self.assertTrue(
        torch.allclose(
            cu_seqlen_q_xla, cu_seqlens_q, rtol=1e-3, atol=1e-3,
            equal_nan=True))
    self.assertTrue(
        torch.allclose(
            cu_seqlen_k_xla, cu_seqlens_k, rtol=1e-3, atol=1e-3,
            equal_nan=True))
    for i in range(len(cu_seq_lens[0]) - 1):
      seqlen = cu_seq_lens[0][i + 1] - cu_seq_lens[0][i]
      self.assertTrue(
          torch.allclose(
              softmax_lse_xla[i, :, :seqlen],
              softmax_lse[i, :, :seqlen],
              rtol=1e-2,
              atol=1e-3,
              equal_nan=True))

  def test_flash_attn_bwd(self):
    d = 32
    batch_size = 4
    nheads = 9
    nheads_k = nheads
    window_size = (-1, -1)

    seqlen_q = 2048
    seqlen_k = 2048

    device = "cuda"
    dtype = torch.bfloat16

    alibi = False
    dropout_p = 0.0
    causal = True
    deterministic = False

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    softmax_scale = q.shape[-1]**(-0.5)
    k = torch.randn(
        batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(
        batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    do = torch.randn(
        batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    rng_state = torch.Tensor([0, 0]).to(torch.int64).to(device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    attention_mask = torch.zeros(
        batch_size, seqlen_k, dtype=torch.int32).to(device)
    k_lengths = torch.randint(low=2, high=seqlen_k, size=(batch_size,))
    for i in range(batch_size):
      k_len = k_lengths[i].item()
      attention_mask[i, :k_len] = 1
      q[i, k_len:, :, :] = 0
      k[i, k_len:, :, :] = 0
      v[i, k_len:, :, :] = 0
      do[i, k_len:, :, :] = 0
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    q_cuda, k_cuda, v_cuda, do_cuda, dq_cuda, dk_cuda, dv_cuda, \
    indices_q, indices_k, cu_seq_lens, max_seq_lens = _bwd_unpad_input(
        q, k, v, do, dq, dk, dv, attention_mask, seqlen_q, nheads
    )
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_q, max_seqlen_k = max_seq_lens

    if alibi:
      alibi_slopes = torch.rand(
          batch_size, nheads, device=device, dtype=torch.float32) * 0.3
    else:
      alibi_slopes = None

    o_cuda, softmax_lse_cuda, _ = flash_attn_varlen_func(
        q_cuda.contiguous(),
        k_cuda.contiguous(),
        v_cuda.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    dq_cuda, dk_cuda, dv_cuda, softmax_d_cuda = flash_attn_cuda.varlen_bwd(
        do_cuda.contiguous(), q_cuda.contiguous(), k_cuda.contiguous(),
        v_cuda.contiguous(), o_cuda.contiguous(), softmax_lse_cuda.contiguous(),
        dq_cuda, dk_cuda, dv_cuda, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal,
        window_size[0], window_size[1], deterministic, None, rng_state)

    dq_cuda = pad_input(dq_cuda, indices_q, batch_size, seqlen_q)
    dk_cuda = pad_input(dk_cuda, indices_k, batch_size, seqlen_k)
    dv_cuda = pad_input(dv_cuda, indices_k, batch_size, seqlen_k)
    softmax_d_cuda = softmax_d_cuda[:, :, :seqlen_q]

    q = q.cpu().detach()
    k = k.cpu().detach()
    v = v.cpu().detach()
    do = do.cpu().detach()
    rng_state = rng_state.cpu().detach()

    dq_cuda = dq_cuda.cpu().detach()
    dk_cuda = dk_cuda.cpu().detach()
    dv_cuda = dv_cuda.cpu().detach()
    softmax_d_cuda = softmax_d_cuda.cpu().detach()
    if alibi:
      alibi_slopes = alibi_slopes.cpu()
    torch.cuda.synchronize()

    device = XLA_DEVICE
    torch.random.manual_seed(101)
    q_xla = q.to(device)
    k_xla = k.to(device)
    v_xla = v.to(device)
    do_xla = do.to(device)
    attention_mask_xla = attention_mask.to(device)
    rng_state_xla = rng_state.to(device)
    if alibi:
      alibi_slopes = alibi_slopes.cpu().to(device)

    _mark_dynamic(q_xla, [1], [seqlen_q + 100])
    _mark_dynamic(k_xla, [1], [seqlen_k + 100])
    _mark_dynamic(v_xla, [1], [seqlen_k + 100])
    _mark_dynamic(attention_mask_xla, [1], [seqlen_k + 100])
    _mark_dynamic(do_xla, [1], [seqlen_q + 100])

    softmax_lse_xla, o_xla, _, cu_seqlen_q_xla, cu_seqlen_k_xla = torch_xla._XLAC._flash_attention_forward(
        q_xla.contiguous(), k_xla.contiguous(), v_xla.contiguous(),
        attention_mask_xla.contiguous(), alibi_slopes, dropout_p, softmax_scale,
        False, causal, window_size[0], window_size[1], True, None)
    q_xla.requires_grad = True
    k_xla.requires_grad = True
    v_xla.requires_grad = True
    o_xla.requires_grad = True
    softmax_lse_xla.requires_grad = True

    dq_xla, dk_xla, dv_xla, softmax_d_xla = torch_xla._XLAC._flash_attention_backward(
        do_xla.contiguous(), q_xla.contiguous(), k_xla.contiguous(),
        v_xla.contiguous(), o_xla.contiguous(), softmax_lse_xla.contiguous(),
        cu_seqlen_q_xla, cu_seqlen_k_xla, alibi_slopes, dropout_p,
        softmax_scale, False, causal, window_size[0], window_size[1],
        deterministic, None, rng_state_xla)

    self.assertIn(f'{batch_size}, <={seqlen_q+100}, {nheads}, {d}',
                  str(o_xla.shape))
    self.assertIn(f'{batch_size}, {nheads}, <={seqlen_q+100}',
                  str(softmax_lse_xla.shape))
    self.assertIn(f'{batch_size}, <={seqlen_q+100}, {nheads}, {d}',
                  str(dq_xla.shape))
    self.assertIn(f'{batch_size}, <={seqlen_q+100}, {nheads}, {d}',
                  str(dk_xla.shape))
    self.assertIn(f'{batch_size}, <={seqlen_q+100}, {nheads}, {d}',
                  str(dv_xla.shape))
    self.assertIn(f'{batch_size}, {nheads}, <={seqlen_q+100}',
                  str(softmax_d_xla.shape))

    xm.mark_step(wait=True)
    torch.cuda.synchronize()

    dq_xla = dq_xla.cpu().detach()
    dk_xla = dk_xla.cpu().detach()
    dv_xla = dv_xla.cpu().detach()
    do_xla = do_xla.cpu().detach()
    softmax_d_xla = softmax_d_xla.cpu().detach()

    self.assertTrue(
        torch.allclose(dq_cuda, dq_xla, rtol=1e-2, atol=1e-2, equal_nan=True))
    self.assertTrue(
        torch.allclose(dk_cuda, dk_xla, rtol=1e-2, atol=1e-2, equal_nan=True))
    self.assertTrue(
        torch.allclose(dv_cuda, dv_xla, rtol=1e-2, atol=1e-2, equal_nan=True))


if __name__ == '__main__':
  assert test_utils.is_disc_backend()
  os.environ['USE_BOUND_FOR_SHAPE_COMPARE'] = os.getenv(
      'USE_BOUND_FOR_SHAPE_COMPARE', '1')
  test = unittest.main()
  # DISABLE PYTHON DISPATCHER FLAG
  del PD
  sys.exit(0 if test.result.wasSuccessful() else 1)
