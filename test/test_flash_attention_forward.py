import sys
import unittest

from flash_attn import flash_attn_varlen_func
import numpy as np
import torch

import torch_xla
import torch_xla.core.xla_model as xm

BATCH_SIZE = 64
SEQ_LEN = 256
DIMS = 16
N_HEADS = 8
N_HEADS_KV = 4
DROPOUT = 0.8
SOFTMAX_SCALE = 0.25
ZERO_TENSORS = False
IS_CAUSAL = True
RETURN_SOFTMAX = True
NUM_SPLITS = 0
GEN = None
WINDOW_SIZE = (-1, -1)
DETERMINISTIC = False


class FlashAttentionForwardTest(unittest.TestCase):

  def _forward_internal(self, tensor_dtype, enable_alibi_slopes=False):
    # init a cuda tensor to initialize torch random generator
    torch.manual_seed(101)
    t1 = torch.zeros([2, 2], device='cuda:0')

    # original flash attention
    device = 'cuda:0'
    q_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    k_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS_KV * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS_KV,
                               DIMS).to(tensor_dtype)
    v_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS_KV * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS_KV,
                               DIMS).to(tensor_dtype)
    out_cuda = torch.zeros_like(q_cuda)
    cu_seqlens_q_cuda = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN,
        step=SEQ_LEN,
        dtype=torch.int32,
        device=device)
    cu_seqlens_k_cuda = cu_seqlens_q_cuda
    alibi_slopes_cuda = torch.linspace(
        -0.5, 0.5, N_HEADS, device=device,
        dtype=torch.float32) * 0.3 if enable_alibi_slopes else None
    out_cuda, softmax_lse_cuda, _ = flash_attn_varlen_func(
        q_cuda,
        k_cuda,
        v_cuda,
        cu_seqlens_q_cuda,
        cu_seqlens_k_cuda,
        SEQ_LEN,
        SEQ_LEN,
        dropout_p=DROPOUT,
        softmax_scale=SOFTMAX_SCALE,
        causal=IS_CAUSAL,
        window_size=WINDOW_SIZE,  # -1 means infinite context window
        alibi_slopes=alibi_slopes_cuda,
        deterministic=DETERMINISTIC,  # always False inside `flash_attn_interface._flash_attn_varlen_forward`
        return_attn_probs=True)
    torch.cuda.synchronize()

    # TorchXLA flash attention
    torch.manual_seed(101)
    t1 = torch.zeros([2, 2], device='cuda:0')
    device = torch_xla.core.xla_model.xla_device()
    q_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    k_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS_KV * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS_KV,
                               DIMS).to(tensor_dtype)
    v_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS_KV * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS_KV,
                               DIMS).to(tensor_dtype)
    cu_seqlens_q_xla = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN,
        step=SEQ_LEN,
        dtype=torch.int32,
        device=device)
    cu_seqlens_k_xla = cu_seqlens_q_xla
    alibi_slopes_xla = torch.linspace(
        -0.5, 0.5, N_HEADS, device=device,
        dtype=torch.float32) * 0.3 if enable_alibi_slopes else None
    softmax_lse_xla, out_xla, _ = torch_xla._XLAC._flash_attention_forward(
        q_xla, k_xla, v_xla, cu_seqlens_q_xla, cu_seqlens_k_xla,
        alibi_slopes_xla, SEQ_LEN, SEQ_LEN, DROPOUT, SOFTMAX_SCALE,
        ZERO_TENSORS, IS_CAUSAL, WINDOW_SIZE[0], WINDOW_SIZE[1], RETURN_SOFTMAX,
        GEN)
    xm.mark_step()
    torch.cuda.synchronize()

    assert torch.allclose(
        softmax_lse_cuda.cpu().detach(),
        softmax_lse_xla.cpu().detach(),
        rtol=1e-2,
        atol=1e-2)
    assert torch.allclose(
        out_cuda.cpu().detach(), out_xla.cpu().detach(), rtol=1e-2, atol=1e-2)

  def test_flash_attn_forward(self):
    self._forward_internal(torch.float16, enable_alibi_slopes=False)
    self._forward_internal(torch.bfloat16, enable_alibi_slopes=False)
    self._forward_internal(torch.float16, enable_alibi_slopes=True)
    self._forward_internal(torch.bfloat16, enable_alibi_slopes=True)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
