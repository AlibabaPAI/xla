import sys
import unittest

import numpy as np
import torch
import flash_attn_2_cuda as flash_attn_cuda

import torch_xla
import torch_xla.core.xla_model as xm

BATCH_SIZE = 64
SEQ_LEN = 256
DIMS = 16
N_HEADS = 8
DROPOUT = 0.8
SOFTMAX_SCALE = 0.25
ZERO_TENSORS = False
IS_CAUSAL = False
NUM_SPLITS = 0
GEN = None


class FlashAttentionBackwardTest(unittest.TestCase):

  def _backward_internal(self, tensor_dtype):
    # original flash attention
    device = 'cuda:0'
    q_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dq_cuda = torch.zeros_like(q_cuda)
    k_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dk_cuda = torch.zeros_like(k_cuda)
    v_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dv_cuda = torch.zeros_like(v_cuda)
    out_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dout_cuda = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    softmax_lse_cuda = torch.linspace(
        5, 6, BATCH_SIZE * N_HEADS * SEQ_LEN,
        device=device).reshape(BATCH_SIZE, N_HEADS, SEQ_LEN).to(torch.float32)
    cu_seqlens_q_cuda = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN,
        step=SEQ_LEN,
        dtype=torch.int32,
        device=q_cuda.device)
    cu_seqlens_k_cuda = cu_seqlens_q_cuda
    rng_state_cuda = torch.Tensor([101, 102]).to(torch.int64).to(device)
    rng_state = torch.cuda.get_rng_state()
    dq_cuda, dk_cuda, dv_cuda, softmax_d_cuda = flash_attn_cuda.varlen_bwd(
        dout_cuda, q_cuda, k_cuda, v_cuda, out_cuda, softmax_lse_cuda, dq_cuda,
        dk_cuda, dv_cuda, cu_seqlens_q_cuda, cu_seqlens_k_cuda, SEQ_LEN,
        SEQ_LEN, DROPOUT, SOFTMAX_SCALE, ZERO_TENSORS, IS_CAUSAL, GEN)

    # TorchXLA flash attention
    device = torch_xla.core.xla_model.xla_device()
    q_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dq_xla = torch.zeros_like(q_xla)
    k_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dk_xla = torch.zeros_like(k_xla)
    v_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dv_xla = torch.zeros_like(v_xla)
    out_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    dout_xla = torch.linspace(
        -0.5, 0.5, SEQ_LEN * BATCH_SIZE * N_HEADS * DIMS,
        device=device).reshape(SEQ_LEN * BATCH_SIZE, N_HEADS,
                               DIMS).to(tensor_dtype)
    softmax_lse_xla = torch.linspace(
        5, 6, BATCH_SIZE * N_HEADS * SEQ_LEN,
        device=device).reshape(BATCH_SIZE, N_HEADS, SEQ_LEN).to(torch.float32)
    cu_seqlens_q_xla = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN,
        step=SEQ_LEN,
        dtype=torch.int32,
        device=q_xla.device)
    cu_seqlens_k_xla = cu_seqlens_q_xla
    rng_state_xla = torch.Tensor([101, 102]).to(torch.int64).to(device)
    torch.cuda.set_rng_state(rng_state)
    dq_xla, dk_xla, dv_xla, softmax_d_xla = torch_xla._XLAC._flash_attention_backward(
        dout_xla, q_xla, k_xla, v_xla, out_xla, softmax_lse_xla,
        cu_seqlens_q_xla, cu_seqlens_k_xla, SEQ_LEN, SEQ_LEN, DROPOUT,
        SOFTMAX_SCALE, ZERO_TENSORS, IS_CAUSAL, GEN)
    xm.mark_step()
    torch.cuda.synchronize()

    assert torch.allclose(
        softmax_d_cuda.cpu().detach(),
        softmax_d_xla.cpu().detach(),
        rtol=1e-3,
        atol=1e-3)
    assert torch.allclose(
        dq_cuda.cpu().detach(), dq_xla.cpu().detach(), rtol=1e-2, atol=1e-2)
    assert torch.allclose(
        dk_cuda.cpu().detach(), dk_xla.cpu().detach(), rtol=1e-2, atol=1e-2)
    assert torch.allclose(
        dv_cuda.cpu().detach(), dv_xla.cpu().detach(), rtol=1e-2, atol=1e-2)

  def test_flash_attn_backward(self):
    self._backward_internal(torch.float16)
    self._backward_internal(torch.bfloat16)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
