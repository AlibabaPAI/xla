import os, sys
import unittest

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
import numpy as np
import torch
import torch.nn.functional as F
import pytest
import torch_xla
import torch_xla.core.xla_model as xm
import flash_attn_2_cuda as flash_attn_cuda
import torchacc as ta


def _get_unpad_data(attention_mask):
  seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
  indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
  max_seqlen_in_batch = seqlens_in_batch.max().item()
  cu_seqlens = F.pad(
      torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
  return (
      indices,
      cu_seqlens,
      max_seqlen_in_batch,
  )


def _upad_input(query_layer, key_layer, value_layer, o_layer, do_layer,
                dq_layer, dk_layer, dv_layer, softmax_lse, attention_mask,
                query_length, n_heads):
  indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
      attention_mask)
  batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape  # b, s, h, d

  key_layer = index_first_axis(
      key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
      indices_k  # filter out the key with unmask query
  )
  dk_layer = index_first_axis(
      dk_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
      indices_k  # filter out the key with unmask query
  )
  value_layer = index_first_axis(
      value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                          head_dim), indices_k)
  dv_layer = index_first_axis(
      dv_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
      indices_k)
  if query_length == kv_seq_len:
    query_layer = index_first_axis(
        query_layer.reshape(batch_size * kv_seq_len, n_heads, head_dim),
        indices_k)
    do_layer = index_first_axis(
        do_layer.reshape(batch_size * kv_seq_len, n_heads, head_dim), indices_k)
    o_layer = index_first_axis(
        o_layer.reshape(batch_size * kv_seq_len, n_heads, head_dim), indices_k)
    dq_layer = index_first_axis(
        dq_layer.reshape(batch_size * kv_seq_len, n_heads, head_dim), indices_k)

    cu_seqlens_q = cu_seqlens_k
    max_seqlen_in_batch_q = max_seqlen_in_batch_k
    indices_q = indices_k
  elif query_length == 1:
    max_seqlen_in_batch_q = 1
    cu_seqlens_q = torch.arange(
        batch_size + 1, dtype=torch.int32,
        device=query_layer.device)  # There is a memcpy here, that is very bad.
    indices_q = cu_seqlens_q[:-1]
    query_layer = query_layer.squeeze(1)
    do_layer = do_layer.squeeze(1)
    o_layer = do_layer.squeeze(1)
    # softmax_lse = do_layer.permute(0, 2, 1).squeeze(1)
  else:
    # The -q_len: slice assumes left padding.
    attention_mask = attention_mask[:, -query_length:]
    query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
        query_layer, attention_mask)
    do_layer, _, _, _ = unpad_input(do_layer, attention_mask)
    o_layer, _, _, _ = unpad_input(o_layer, attention_mask)
    dq_layer, _, _, _ = unpad_input(dq_layer, attention_mask)

  softmax_lse = softmax_lse[:, :, :max_seqlen_in_batch_q]
  return (
      query_layer,  # (b*s, h, d), b*s is the true data
      key_layer,  # (b*s, h, d)
      value_layer,  # (b*s, h, d)
      o_layer,
      do_layer,
      dq_layer,
      dk_layer,
      dv_layer,
      softmax_lse,
      indices_q,
      indices_k,
      (cu_seqlens_q, cu_seqlens_k),
      (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
  )


@pytest.fixture(autouse=True, scope="module")
def setup_env():
  orign_env = os.getenv('PJRT_ALLOCATOR_FRACTION')
  os.environ['PJRT_ALLOCATOR_FRACTION'] = '0.5'
  yield
  if orign_env is None:
    os.environ.pop('PJRT_ALLOCATOR_FRACTION', None)
  else:
    os.environ['PJRT_ALLOCATOR_FRACTION'] = orign_env


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["gqa"])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [8])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (8, 8),
        (128, 128),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_backward(seqlen_q, seqlen_k, d, dropout_p, causal, local,
                             alibi, deterministic, mha_type, dtype):
  if d % 8 != 0:
    pytest.skip(reason="Expected head_size_og % 8 == 0 to be true")

  device = "cuda"
  # set seed
  torch.random.manual_seed(101)
  batch_size = 4
  nheads = 9
  nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)

  assert nheads % nheads_k == 0
  window_size = (-1, -1) if not local else tuple(
      torch.randint(0, seqlen_k, (2,)).tolist())
  torch.cuda.synchronize()
  q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
  softmax_scale = q.shape[-1]**(-0.5)
  k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
  v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
  o = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
  do = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
  softmax_lse = torch.randn(
      batch_size, nheads, seqlen_q, device=device, dtype=torch.float32)
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
    o[i, k_len:, :, :] = 0
    do[i, k_len:, :, :] = 0
    softmax_lse[i, :, :k_len] = 0
  q.requires_grad = True
  k.requires_grad = True
  v.requires_grad = True
  o.requires_grad = True
  softmax_lse.requires_grad = True

  q_cuda, k_cuda, v_cuda, o_cuda, do_cuda, dq_cuda, dk_cuda, dv_cuda, \
  softmax_lse_cuda, indices_q, indices_k, cu_seq_lens, max_seq_lens = _upad_input(
      q, k, v, o, do, dq, dk, dv, softmax_lse, attention_mask, seqlen_q, nheads
  )
  cu_seqlens_q, cu_seqlens_k = cu_seq_lens
  max_seqlen_q, max_seqlen_k = max_seq_lens

  if alibi:
    alibi_slopes = torch.rand(
        batch_size, nheads, device=device, dtype=torch.float32) * 0.3
  else:
    alibi_slopes = None
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
  o = o.cpu().detach()
  do = do.cpu().detach()
  rng_state = rng_state.cpu().detach()
  softmax_lse = softmax_lse.cpu().detach()

  dq_cuda = dq_cuda.cpu().detach()
  dk_cuda = dk_cuda.cpu().detach()
  dv_cuda = dv_cuda.cpu().detach()
  softmax_d_cuda = softmax_d_cuda.cpu().detach()
  if alibi:
    alibi_slopes = alibi_slopes.cpu()
  torch.cuda.synchronize()

  device = ta.lazy_device()
  torch.random.manual_seed(101)
  q_xla = q.to(device)
  k_xla = k.to(device)
  v_xla = v.to(device)
  o_xla = o.to(device)
  do_xla = do.to(device)
  softmax_lse_xla = softmax_lse.to(device)
  rng_state_xla = rng_state.to(device)

  dq_xla = dq.to(device)
  dk_xla = dk.to(device)
  dv_xla = dv.to(device)
  softmax_d_xla = softmax_d_cuda.to(device)
  q_xla.requires_grad = True
  k_xla.requires_grad = True
  v_xla.requires_grad = True
  o_xla.requires_grad = True
  softmax_lse_xla.requires_grad = True

  cu_seqlens_q = cu_seqlens_q.to(device)
  cu_seqlens_k = cu_seqlens_k.to(device)
  if alibi:
    alibi_slopes = alibi_slopes.cpu().to(device)
  dq_xla, dk_xla, dv_xla, softmax_d_xla = torch_xla._XLAC._flash_attention_backward(
      do_xla.contiguous(), q_xla.contiguous(), k_xla.contiguous(),
      v_xla.contiguous(), o_xla.contiguous(), softmax_lse_xla.contiguous(),
      cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(), alibi_slopes,
      dropout_p, softmax_scale, False, causal, window_size[0], window_size[1],
      deterministic, None, rng_state_xla)

  ta.mark_step(wait=True)
  torch.cuda.synchronize()
  q_xla = q_xla.cpu().detach()
  k_xla = k_xla.cpu().detach()
  v_xla = v_xla.cpu().detach()
  o_xla = o_xla.cpu().detach()
  dq_xla = dq_xla.cpu().detach()
  dk_xla = dk_xla.cpu().detach()
  dv_xla = dv_xla.cpu().detach()
  do_xla = do_xla.cpu().detach()
  softmax_lse_xla = softmax_lse_xla.cpu().detach()
  softmax_d_xla = softmax_d_xla.cpu().detach()
  rng_state_xla = rng_state_xla.cpu().detach()

  difference_dq = torch.abs(dq_cuda - dq_xla)
  tolerance_dq = 1e-5 + 1e-5 * torch.abs(dq_xla)
  mask_dq = difference_dq > tolerance_dq
  indices_dq = mask_dq.nonzero(as_tuple=False)
  assert (indices_dq.numel() < q.numel() * 1e-2)

  difference_dk = torch.abs(dk_cuda - dk_xla)
  tolerance_dk = 1e-3 + 1e-3 * torch.abs(dk_xla)
  mask_dk = difference_dk > tolerance_dk
  indices_dk = mask_dk.nonzero(as_tuple=False)
  assert (indices_dk.numel() < q.numel() * 1e-2)

  difference_dv = torch.abs(dv_cuda - dv_xla)
  tolerance_dv = 1e-3 + 1e-3 * torch.abs(dv_xla)
  mask_dv = difference_dv > tolerance_dv
  indices_dv = mask_dv.nonzero(as_tuple=False)
  assert (indices_dv.numel() < q.numel() * 1e-2)

  softmax_d_cuda = softmax_d_cuda[:, :, :max_seqlen_q]
  softmax_d_xla = softmax_d_xla[:, :, :max_seqlen_q]
  difference_ds = torch.abs(softmax_d_cuda - softmax_d_xla)
  tolerance_ds = 1e-3 + 1e-3 * torch.abs(softmax_d_xla)
  mask_ds = difference_ds > tolerance_ds
  indices_ds = mask_ds.nonzero(as_tuple=False)
  assert (indices_ds.numel() < softmax_d_xla.numel() * 1e-2)
