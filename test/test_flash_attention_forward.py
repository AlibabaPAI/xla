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
DROPOUT = 0.8
SOFTMAX_SCALE = 0.25
ZERO_TENSORS = True
IS_CAUSAL = False
RETURN_SOFTMAX = False
NUM_SPLITS = 0
GEN = None


class FlashAttentionForwardTest(unittest.TestCase):

  def _forward_internal(self, tensor_dtype):
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
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    v_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    out_cuda = torch.zeros_like(q_cuda)
    cu_seqlens_q_cuda = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN,
        step=SEQ_LEN,
        dtype=torch.int32,
        device=device)
    cu_seqlens_k_cuda = cu_seqlens_q_cuda
    out_cuda, softmax_lse_cuda, S_dmask_cuda = flash_attn_varlen_func(
        q_cuda, k_cuda, v_cuda, cu_seqlens_q_cuda, cu_seqlens_k_cuda, SEQ_LEN,
        SEQ_LEN, DROPOUT, SOFTMAX_SCALE, IS_CAUSAL, RETURN_SOFTMAX)
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
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    v_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    cu_seqlens_q_xla = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN,
        step=SEQ_LEN,
        dtype=torch.int32,
        device=device)
    cu_seqlens_k_xla = cu_seqlens_q_xla
    softmax_lse_xla, out_xla = torch_xla._XLAC._flash_attention_forward(
        q_xla, k_xla, v_xla, cu_seqlens_q_xla, cu_seqlens_k_xla, SEQ_LEN,
        SEQ_LEN, DROPOUT, SOFTMAX_SCALE, ZERO_TENSORS, IS_CAUSAL,
        RETURN_SOFTMAX, GEN)
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
    self._forward_internal(torch.float16)
    self._forward_internal(torch.bfloat16)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
