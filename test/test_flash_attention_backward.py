import os
import pytest

import torch
import torch_xla

from flash_attn import flash_attn_func
import flash_attn_2_cuda as flash_attn_cuda
import torchacc as ta


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
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 113),
        (128, 128),
        (256, 256),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_backward(seqlen_q, seqlen_k, d, dropout_p, causal, local,
                             alibi, deterministic, mha_type, dtype):
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
      requires_grad=True)
  softmax_scale = q.shape[-1]**(-0.5)
  k = torch.randn(
      batch_size,
      seqlen_k,
      nheads_k,
      d,
      device=device,
      dtype=dtype,
      requires_grad=True)
  v = torch.randn(
      batch_size,
      seqlen_k,
      nheads_k,
      d,
      device=device,
      dtype=dtype,
      requires_grad=True)
  do = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

  rng_state = torch.Tensor([0, 0]).to(torch.int64).to(device)
  dq = torch.empty_like(q)
  dk = torch.empty_like(k)
  dv = torch.empty_like(v)

  if alibi:
    alibi_slopes = torch.rand(
        batch_size, nheads, device=device, dtype=torch.float32) * 0.3
  else:
    alibi_slopes = None

  o, softmax_lse, _ = flash_attn_func(
      q,
      k,
      v,
      dropout_p,
      softmax_scale=softmax_scale,
      causal=causal,
      window_size=window_size,
      alibi_slopes=alibi_slopes,
      deterministic=deterministic,
      return_attn_probs=True,
  )

  dq, dk, dv, softmax_d = flash_attn_cuda.bwd(do, q, k, v, o, softmax_lse, dq,
                                              dk, dv, alibi_slopes, dropout_p,
                                              softmax_scale, causal,
                                              window_size[0], window_size[1],
                                              deterministic, None, rng_state)

  torch.random.manual_seed(0)
  q = q.cpu().detach()
  k = k.cpu().detach()
  v = v.cpu().detach()
  o = o.cpu().detach()
  do = do.cpu().detach()
  rng_state = rng_state.cpu().detach()
  softmax_lse = softmax_lse.cpu().detach()

  dq = dq.cpu().detach()
  dk = dk.cpu().detach()
  dv = dv.cpu().detach()
  softmax_d = softmax_d.cpu().detach()
  torch.cuda.synchronize()

  device = ta.lazy_device()
  torch.random.manual_seed(0)
  q_xla = q.to(device)
  k_xla = k.to(device)
  v_xla = v.to(device)
  do_xla = do.to(device)

  softmax_d_xla = softmax_d.to(device)
  q_xla.requires_grad = True
  k_xla.requires_grad = True
  v_xla.requires_grad = True
  if alibi:
    alibi_slopes = alibi_slopes.cpu().to(device)
  softmax_lse_xla, o_xla, rng_state_xla = torch_xla._XLAC._flash_attention_forward(
      q_xla, k_xla, v_xla, None, alibi_slopes, dropout_p, softmax_scale, False,
      causal, window_size[0], window_size[1], True, None)

  dq_xla, dk_xla, dv_xla, softmax_d_xla = torch_xla._XLAC._flash_attention_backward(
      do_xla, q_xla, k_xla, v_xla, o_xla, softmax_lse_xla, None, None,
      alibi_slopes, dropout_p, softmax_scale, False, causal, window_size[0],
      window_size[1], deterministic, None, rng_state_xla)

  ta.mark_step(wait=True)
  torch.cuda.synchronize()

  dq_xla = dq_xla.cpu().detach()
  dk_xla = dk_xla.cpu().detach()
  dv_xla = dv_xla.cpu().detach()
  softmax_d_xla = softmax_d_xla.cpu().detach()

  assert torch.allclose(dq, dq_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(dk, dk_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(dv, dv_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
  assert torch.allclose(
      softmax_d, softmax_d_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
