import os
import pytest

import torch
import torch.nn.functional as F
import torch_xla

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
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


def _unpad_input(query_layer, key_layer, value_layer, do_layer, dq_layer,
                 dk_layer, dv_layer, attention_mask, query_length, n_heads):
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
    dq_layer = index_first_axis(
        dq_layer.reshape(batch_size * kv_seq_len, n_heads, head_dim), indices_k)

    cu_seqlens_q = cu_seqlens_k
    max_seqlen_in_batch_q = max_seqlen_in_batch_k
    indices_q = indices_k
  elif query_length == 1:
    max_seqlen_in_batch_q = 1
    cu_seqlens_q = torch.arange(
        batch_size + 1, dtype=torch.int32, device=query_layer.device)
    indices_q = cu_seqlens_q[:-1]
    query_layer = query_layer.squeeze(1)
    do_layer = do_layer.squeeze(1)
  else:
    # The -q_len: slice assumes left padding.
    attention_mask = attention_mask[:, -query_length:]
    query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
        query_layer, attention_mask)
    do_layer, _, _, _ = unpad_input(do_layer, attention_mask)
    dq_layer, _, _, _ = unpad_input(dq_layer, attention_mask)

  return (
      query_layer,  # (total_q, h, d),
      key_layer,  # (total_k, h, d)
      value_layer,  # (total_k, h, d)
      do_layer,
      dq_layer,
      dk_layer,
      dv_layer,
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [8])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_varlen_backward(seqlen_q, seqlen_k, d, dropout_p, causal,
                                    local, alibi, deterministic, mha_type,
                                    dtype):
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
  do = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
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
  indices_q, indices_k, cu_seq_lens, max_seq_lens = _unpad_input(
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

  device = ta.lazy_device()
  torch.random.manual_seed(101)
  q_xla = q.to(device)
  k_xla = k.to(device)
  v_xla = v.to(device)
  do_xla = do.to(device)
  attention_mask_xla = attention_mask.to(device)
  rng_state_xla = rng_state.to(device)
  if alibi:
    alibi_slopes = alibi_slopes.cpu().to(device)

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
      cu_seqlen_q_xla, cu_seqlen_k_xla, alibi_slopes, dropout_p, softmax_scale,
      False, causal, window_size[0], window_size[1], deterministic, None,
      rng_state_xla)

  ta.mark_step(wait=True)
  torch.cuda.synchronize()

  dq_xla = dq_xla.cpu().detach()
  dk_xla = dk_xla.cpu().detach()
  dv_xla = dv_xla.cpu().detach()
  do_xla = do_xla.cpu().detach()
  softmax_d_xla = softmax_d_xla.cpu().detach()

  assert torch.allclose(dq_cuda, dq_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(dk_cuda, dk_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(dv_cuda, dv_xla, rtol=1e-2, atol=1e-2, equal_nan=True)




@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [8])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_varlen_position_ids_backward(seqlen_q, seqlen_k, d, dropout_p, causal,
                                    local, alibi, deterministic, mha_type,
                                    dtype):
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
  do = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
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
  indices_q, indices_k, cu_seq_lens, max_seq_lens = _unpad_input(
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

  device = ta.lazy_device()
  torch.random.manual_seed(101)
  
  indices_q = indices_q.cpu()
  indices_k = indices_k.cpu()

  q_xla = q.flatten(0,1)[indices_q].unsqueeze(0).to(device)
  k_xla = k.flatten(0,1)[indices_k].unsqueeze(0).to(device)
  v_xla = v.flatten(0,1)[indices_k].unsqueeze(0).to(device)
  do_xla = do.flatten(0,1)[indices_q].unsqueeze(0).to(device)
  
  def attention_mask_to_position_ids(attention_mask):
    seqlens = attention_mask.sum(dim=1).flatten()
    position_ids = torch.cat([torch.arange(0,seqlen,dtype=torch.int32,device=attention_mask.device) for seqlen in seqlens],dim=0)
    return position_ids.unsqueeze(0)

  position_ids_xla = attention_mask_to_position_ids(attention_mask).to(device)

  rng_state_xla = rng_state.to(device)
  if alibi:
    alibi_slopes = alibi_slopes.cpu().to(device)
  

  softmax_lse_xla, o_xla, _, cu_seqlen_q_xla, cu_seqlen_k_xla = torch_xla._XLAC._flash_attention_position_ids_forward(
      q_xla.contiguous(), k_xla.contiguous(), v_xla.contiguous(),
      position_ids_xla.contiguous(), alibi_slopes, dropout_p, softmax_scale,
      False, causal, window_size[0], window_size[1], True, None)

  assert torch.allclose(q_xla.cpu(), q_cuda.cpu().unsqueeze(0), rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(k_xla.cpu(), k_cuda.cpu().unsqueeze(0), rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(v_xla.cpu(), v_cuda.cpu().unsqueeze(0), rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(
      cu_seqlen_k_xla[:batch_size+1].cpu(), cu_seqlens_k.cpu(), rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(o_xla.cpu(), o_cuda.unsqueeze(0).cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

  q_xla.requires_grad = True
  k_xla.requires_grad = True
  v_xla.requires_grad = True
  o_xla.requires_grad = True
  softmax_lse_xla.requires_grad = True

  dq_xla, dk_xla, dv_xla, softmax_d_xla = torch_xla._XLAC._flash_attention_position_ids_backward(
      do_xla.contiguous(), q_xla.contiguous(), k_xla.contiguous(),
      v_xla.contiguous(), o_xla.contiguous(), softmax_lse_xla.contiguous(),
      cu_seqlen_q_xla, cu_seqlen_k_xla, alibi_slopes, dropout_p, softmax_scale,
      False, causal, window_size[0], window_size[1], deterministic, None,
      rng_state_xla)

  ta.mark_step(wait=True)
  torch.cuda.synchronize()

  dq_xla = dq_xla.cpu().detach().squeeze(0)
  dk_xla = dk_xla.cpu().detach().squeeze(0)
  dv_xla = dv_xla.cpu().detach().squeeze(0)
  do_xla = do_xla.cpu().detach().squeeze(0)
  softmax_d_xla = softmax_d_xla.cpu().detach()
  
  dq_xla = pad_input(dq_xla, indices_q, batch_size, seqlen_q)
  dk_xla = pad_input(dk_xla, indices_k, batch_size, seqlen_k)
  dv_xla = pad_input(dv_xla, indices_k, batch_size, seqlen_k)

  assert torch.allclose(dq_cuda, dq_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(dk_cuda, dk_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(dv_cuda, dv_xla, rtol=1e-2, atol=1e-2, equal_nan=True)

