import os
import pytest

import torch
import torch.nn.functional as F
import torch_xla

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
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


def _unpad_input(query_layer, key_layer, value_layer, attention_mask,
                 query_length, n_heads):
  indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
      attention_mask)
  batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape  # b, s, h, d

  key_layer = index_first_axis(
      key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
      indices_k  # filter out the key with unmask query
  )
  value_layer = index_first_axis(
      value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                          head_dim), indices_k)
  if query_length == kv_seq_len:
    query_layer = index_first_axis(
        query_layer.reshape(batch_size * kv_seq_len, n_heads, head_dim),
        indices_k)
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
  else:
    # The -q_len: slice assumes left padding.
    attention_mask = attention_mask[:, -query_length:]
    query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
        query_layer, attention_mask)

  return (
      query_layer,
      key_layer,
      value_layer,
      indices_q,
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
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("softmax_scale", [0.25])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (8, 8),
        (128, 128),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_output(seqlen_q, seqlen_k, d, dropout_p, causal,
                           softmax_scale, local, alibi, deterministic, mha_type,
                           dtype):
  # (TODO: wenting.swt) When run all test cases together, some cases show a precision
  # difference > 1e-2 at a few positions; however, these cases can run successfully
  # when run individually. The cause of this issue needs to be identified.
  if d % 8 != 0:
    pytest.skip(reason="Expected head_size_og % 8 == 0 to be true")

  device = "cuda"
  # set seed
  torch.random.manual_seed(0)
  batch_size = 4
  nheads = 9
  nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)

  assert nheads % nheads_k == 0
  window_size = (-1, -1) if not local else tuple(
      torch.randint(0, seqlen_k, (2,)).tolist())
  torch.cuda.synchronize()
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
  q_cuda, k_cuda, v_cuda, indices_q, cu_seq_lens, max_seq_lens = _unpad_input(
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

  device = ta.lazy_device()
  torch.random.manual_seed(0)
  q_xla = q.to(device)
  k_xla = k.to(device)
  v_xla = v.to(device)
  attention_mask_xla = attention_mask.to(device)
  q_xla.requires_grad = False
  k_xla.requires_grad = False
  v_xla.requires_grad = False
  if alibi:
    alibi_slopes = alibi_slopes.cpu().to(device)
  softmax_lse_xla, out_xla, _, cu_seqlen_q_xla, cu_seqlen_k_xla = torch_xla._XLAC._flash_attention_forward(
      q_xla.contiguous(), k_xla.contiguous(), v_xla.contiguous(),
      attention_mask_xla.contiguous(), alibi_slopes, dropout_p, softmax_scale,
      False, causal, window_size[0], window_size[1], True, None)

  ta.mark_step(wait=True)
  q_xla = q_xla.cpu().detach()
  k_xla = k_xla.cpu().detach()
  v_xla = v_xla.cpu().detach()
  out_xla = out_xla.cpu().detach()
  cu_seqlen_q_xla = cu_seqlen_q_xla.cpu().detach()
  cu_seqlen_k_xla = cu_seqlen_k_xla.cpu().detach()
  softmax_lse_xla = softmax_lse_xla.cpu().detach()
  attention_mask_xla = attention_mask_xla.cpu().detach()

  assert torch.allclose(q_xla, q, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(k_xla, k, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(v_xla, v, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(out_xla, out_fa, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(
      cu_seqlen_q_xla, cu_seqlens_q, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(
      cu_seqlen_k_xla, cu_seqlens_k, rtol=1e-3, atol=1e-3, equal_nan=True)
  for i in range(len(cu_seq_lens[0]) - 1):
    seqlen = cu_seq_lens[0][i + 1] -  cu_seq_lens[0][i]
    assert torch.allclose(softmax_lse_xla[i,:,:seqlen],softmax_lse[i,:,:seqlen],rtol=1e-2,atol=1e-3,equal_nan=True)  
  




@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("softmax_scale", [0.25])
@pytest.mark.parametrize(
    "max_seqlen_q,max_seqlen_k",
    [
        (8, 8),
        (128, 128),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_varlen_from_position_ids(max_seqlen_q, max_seqlen_k, d, dropout_p, causal,
                           softmax_scale, local, alibi, deterministic, mha_type,
                           dtype):
  if d % 8 != 0:
    pytest.skip(reason="Expected head_size_og % 8 == 0 to be true")

  device = "cuda"
  # set seed
  torch.random.manual_seed(0)
  batch_size = 4
  nheads = 9
  nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3) 

  assert nheads % nheads_k == 0
  window_size = (-1, -1) if not local else tuple(
      torch.randint(0, max_seqlen_k, (2,)).tolist())
  torch.cuda.synchronize()

  def generate_qkv_and_position_ids(batch_size,max_seqlen_q, max_seqlen_k, dtype, n_heads_q, n_heads_k, device, head_dim, use_same_seqlen):
    '''
    generate varlen qkv and postion_ids and pad total seqlen to 8
    '''
    seq_len_q = torch.randint(1,max_seqlen_q+1,(batch_size,))
    if use_same_seqlen:
      seq_len_k = seq_len_q.clone()
    else:
      seq_len_k = torch.randint(1,max_seqlen_k+1,(batch_size,))
    total_q = seq_len_q.sum().item()
    total_k = seq_len_k.sum().item()
    
    padd_q = 0 if total_q % 8 == 0 else 8 - total_q % 8
    padd_k = 0 if total_k % 8 == 0 else 8 - total_k % 8
    if use_same_seqlen:
      assert torch.all(seq_len_k == seq_len_q)

    # padding to last q and k
    if padd_q:
      seq_len_q[-1] += padd_q
      total_q += padd_q
    assert total_q % 8 == 0
    if padd_k:
      seq_len_k[-1] += padd_k
      total_k += padd_k
    assert total_k % 8 == 0

    q = torch.randn((1,total_q,n_heads_q,head_dim),dtype=dtype,device=device)
    k = torch.randn((1,total_k,n_heads_k,head_dim),dtype=dtype,device=device)
    v = torch.randn((1,total_k,n_heads_k,head_dim),dtype=dtype,device=device)

    assert torch.all(seq_len_q > 0)
    assert torch.all(seq_len_k > 0) 

    position_ids_q = torch.cat([torch.arange(0,seq_len,dtype=torch.int32,device=device) for seq_len in seq_len_q],dim=0).unsqueeze(0)
    position_ids_k = torch.cat([torch.arange(0,seq_len,dtype=torch.int32,device=device) for seq_len in seq_len_k],dim=0).unsqueeze(0)
    assert position_ids_q.shape[1] % 8 == 0
    assert position_ids_k.shape[1] % 8 == 0
    
    return q,k,v,position_ids_q,position_ids_k
  

  q,k,v,position_ids_q,position_ids_k = generate_qkv_and_position_ids(batch_size,max_seqlen_q,max_seqlen_k,dtype,nheads,nheads_k,device,d,max_seqlen_q == max_seqlen_k)
 
  indices_q = torch.arange(0,position_ids_q.size(1),device=device,dtype=torch.int64)
  indices_k = torch.arange(0,position_ids_k.size(1),device=device,dtype=torch.int64)

  seq_q_start_idx = indices_q[position_ids_q.squeeze() == 0]
  seq_k_start_idx = indices_k[position_ids_k.squeeze() == 0]
  
  cu_seq_lens_q = F.pad(seq_q_start_idx,(0,1),value=position_ids_q.size(1)).to(torch.int32)
  cu_seq_lens_k = F.pad(seq_k_start_idx,(0,1),value=position_ids_k.size(1)).to(torch.int32)

  max_seqlen_q = (cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]).max().item()
  max_seqlen_k = (cu_seq_lens_k[1:] - cu_seq_lens_k[:-1]).max().item()

  q_cuda = q.reshape(-1,nheads,d).cuda()
  k_cuda = k.reshape(-1,nheads_k,d).cuda()
  v_cuda = v.reshape(-1,nheads_k,d).cuda()


  if alibi:
    alibi_slopes = torch.rand(
        batch_size, nheads, device=device, dtype=torch.float32) * 0.3
  else:
    alibi_slopes = None
  

  out_fa, softmax_lse, _ = flash_attn_varlen_func(
      q_cuda.contiguous(),
      k_cuda.contiguous(),
      v_cuda.contiguous(),
      cu_seq_lens_q.contiguous(),
      cu_seq_lens_k.contiguous(),
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

  out_fa = pad_input(out_fa, indices_q, batch_size, max_seqlen_q)

  q = q.cpu().detach()
  k = k.cpu().detach()
  v = v.cpu().detach()

  out_fa = out_fa.cpu().detach()
  softmax_lse = softmax_lse.cpu().detach()
  cu_seq_lens_q = cu_seq_lens_q.cpu().detach()
  cu_seq_lens_k = cu_seq_lens_k.cpu().detach()

  if alibi:
    alibi_slopes = alibi_slopes.cpu()
  torch.cuda.synchronize()

  device = ta.lazy_device()
  torch.random.manual_seed(0)
  q_xla = q.to(device)
  k_xla = k.to(device)
  v_xla = v.to(device)

  position_ids_q_xla = position_ids_q.to(device)

  q_xla.requires_grad = False
  k_xla.requires_grad = False
  v_xla.requires_grad = False
  if alibi:
    alibi_slopes = alibi_slopes.cpu().to(device)

  softmax_lse_xla, out_xla, _, cu_seq_len_q_xla, cu_seq_len_k_xla = torch_xla._XLAC._flash_attention_position_ids_forward(
      q_xla.contiguous(), k_xla.contiguous(), v_xla.contiguous(),
      position_ids_q_xla.contiguous(), alibi_slopes, dropout_p, softmax_scale,
      False, causal, window_size[0], window_size[1], True, None)


  ta.mark_step(wait=True)
  q_xla = q_xla.cpu().detach()
  k_xla = k_xla.cpu().detach()
  v_xla = v_xla.cpu().detach()
  out_xla = out_xla.cpu().detach()
  cu_seq_len_q_xla = cu_seq_len_q_xla.cpu().detach()
  cu_seq_len_k_xla = cu_seq_len_k_xla.cpu().detach()
  softmax_lse_xla = softmax_lse_xla.cpu().detach()
  position_ids_q_xla = position_ids_q_xla.cpu().detach()

  out_xla = out_xla.squeeze(0)

  out_xla = pad_input(out_xla, indices_q, batch_size, max_seqlen_q)

  assert out_xla.shape == out_fa.shape

  assert torch.allclose(q_xla, q, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(k_xla, k, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(v_xla, v, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(
      cu_seq_len_k_xla[:batch_size+1], cu_seq_lens_k, rtol=1e-3, atol=1e-3, equal_nan=True)
  assert torch.allclose(out_xla, out_fa, rtol=1e-2, atol=1e-2, equal_nan=True)
  for i in range(batch_size):
    start_idx = cu_seq_len_q_xla[i]
    end_idx = cu_seq_len_q_xla[i+1]
    seqlen = end_idx - start_idx
    assert torch.allclose(softmax_lse_xla[0,:,start_idx:end_idx],softmax_lse[i,:,:seqlen],rtol=1e-3,atol=1e-3,equal_nan=True)  
