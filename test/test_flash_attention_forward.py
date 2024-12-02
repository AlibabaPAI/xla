import os
import pytest

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from flash_attn import flash_attn_func
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
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [256])
@pytest.mark.parametrize("softmax_scale", [0.25])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (118, 127),
        (128, 128),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_output(seqlen_q, seqlen_k, d, dropout_p, causal,
                           softmax_scale, local, alibi, deterministic, mha_type,
                           dtype):
  if d % 8 != 0:
    pytest.skip(reason="Expected head_size_og % 8 == 0 to be true")

  device = "cuda"
  # set seed
  torch.random.manual_seed(100)
  batch_size = 1
  nheads = 9
  nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)

  assert nheads % nheads_k == 0
  window_size = (-1, -1) if not local else tuple(
      torch.randint(0, seqlen_k, (2,)).tolist())
  torch.cuda.synchronize()
  q_cuda = torch.randn(
      seqlen_q,
      batch_size,
      nheads,
      d,
      device=device,
      dtype=dtype,
      requires_grad=True)
  k_cuda = torch.randn(
      seqlen_k,
      batch_size,
      nheads_k,
      d,
      device=device,
      dtype=dtype,
      requires_grad=True)
  v_cuda = torch.randn(
      seqlen_k,
      batch_size,
      nheads_k,
      d,
      device=device,
      dtype=dtype,
      requires_grad=True)

  softmax_scale = q_cuda.shape[-1]**(-0.5)

  if alibi:
    alibi_slopes = torch.rand(
        batch_size, nheads, device=device, dtype=torch.float32) * 0.3
  else:
    alibi_slopes = None
  q = q_cuda.transpose(0, 1)
  k = k_cuda.transpose(0, 1)
  v = v_cuda.transpose(0, 1)
  out_fa, softmax_lse, _ = flash_attn_func(
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
  q = q.cpu().detach()
  k = k.cpu().detach()
  v = v.cpu().detach()
  out_fa = out_fa.cpu().detach()
  softmax_lse = softmax_lse.cpu().detach()
  if alibi:
    alibi_slopes = alibi_slopes.cpu()
  torch.cuda.synchronize()

  device = ta.lazy_device()
  torch.random.manual_seed(100)
  q_xla = q_cuda.to(device)
  k_xla = k_cuda.to(device)
  v_xla = v_cuda.to(device)
  q_xla = q_xla.transpose(0, 1)
  k_xla = k_xla.transpose(0, 1)
  v_xla = v_xla.transpose(0, 1)
  if alibi:
    alibi_slopes = alibi_slopes.cpu().to(device)
  softmax_lse_xla, out_xla, rng_state_xla = torch_xla._XLAC._flash_attention_forward(
      q_xla, k_xla, v_xla, None, alibi_slopes, dropout_p, softmax_scale, False,
      causal, window_size[0], window_size[1], True, None)

  ta.mark_step(wait=True)
  q_xla = q_xla.cpu().detach()
  k_xla = k_xla.cpu().detach()
  v_xla = v_xla.cpu().detach()
  out_xla = out_xla.cpu().detach()

  rng_state_xla = rng_state_xla.cpu().detach()
  softmax_lse_xla = softmax_lse_xla.cpu().detach()

  assert torch.allclose(softmax_lse_xla, softmax_lse, rtol=1e-2, atol=1e-2)
  assert torch.allclose(out_xla, out_fa, rtol=1e-2, atol=1e-2)



