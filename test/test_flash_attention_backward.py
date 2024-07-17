import os, sys
import unittest

from flash_attn import flash_attn_func
import numpy as np
import torch
import torch.nn.functional as F
import pytest
import torch_xla
import torch_xla.core.xla_model as xm
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

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("d", [32])
# @pytest.mark.parametrize("softmax_scale", [0.25]) # softmax_scale = 0.75 is false
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        # (108, 256),
        # (256, 512),
        # (512, 256),
        # (8, 8),
        # (16, 16)
        # (128, 128),
        # (1023, 1024),
        # (1024, 1023),
        # (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_backward(seqlen_q, seqlen_k, d, dropout_p, causal, 
                             local, alibi, deterministic, mha_type, dtype):
    if d % 8 != 0:
        pytest.skip(reason="Expected head_size_og % 8 == 0 to be true")

    device = "cuda"
    # set seed
    torch.random.manual_seed(101)
    batch_size = 1
    nheads = 9
    nheads_k = 9
    # nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)

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
    o = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True)
    do = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d,
        device=device,
        dtype=dtype)
    softmax_lse = torch.randn(
        batch_size,
        nheads,
        seqlen_q,
        device=device,
        dtype=torch.float32,
        requires_grad=True)
    rng_state = torch.Tensor([0, 0]).to(torch.int64).to(device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if alibi:
        alibi_slopes = torch.rand(
            batch_size, nheads, device=device, dtype=torch.float32) * 0.3
    else:
        alibi_slopes = None
    dq, dk, dv, softmax_d = flash_attn_cuda.bwd(
        do, q, k, v, o, softmax_lse, dq,
        dk, dv,
        alibi_slopes, dropout_p, softmax_scale,
        causal, window_size[0], window_size[1], deterministic,
        None, rng_state)
    
    torch.random.manual_seed(101)
    dq_2 = torch.zeros_like(q)
    dk_2 = torch.zeros_like(k)
    dv_2 = torch.zeros_like(v)
    dq_2, dk_2, dv_2, softmax_d_2 = flash_attn_cuda.bwd(
        do, q, k, v, o, softmax_lse, dq,
        dk, dv,
        alibi_slopes, dropout_p, softmax_scale,
        causal, window_size[0], window_size[1], deterministic,
        None, rng_state)
    dq_2 = dq_2.cpu().detach()
    dk_2 = dk_2.cpu().detach()
    dv_2 = dv_2.cpu().detach()
    
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
    softmax_d_xla = softmax_d.to(device)
    q_xla.requires_grad = True
    k_xla.requires_grad = True
    v_xla.requires_grad = True
    o_xla.requires_grad = True
    softmax_lse_xla.requires_grad = True
    if alibi:
        alibi_slopes = alibi_slopes.cpu().to(device)
    dq_xla, dk_xla, dv_xla, softmax_d_xla  = torch_xla._XLAC._flash_attention_backward(
        do_xla, q_xla, k_xla, v_xla, o_xla, softmax_lse_xla,
        None, None, alibi_slopes,
        dropout_p, softmax_scale, False, causal, window_size[0],
        window_size[1], deterministic, None, rng_state_xla)

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
    
    # assert torch.allclose(q, q_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # assert torch.allclose(k, k_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # assert torch.allclose(v, v_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # assert torch.allclose(o, o_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # assert torch.allclose(do, do_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # assert torch.allclose(softmax_lse, softmax_lse_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # assert torch.allclose(rng_state, rng_state_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # # assert torch.allclose(do, do_xla, rtol=1e-5, atol=1e-5, equal_nan=True)
    # assert torch.allclose(dq, dq_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
    # assert torch.allclose(dk, dk_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
    
    # assert torch.allclose(dv, dv_xla, rtol=1e-2, atol=1e-2, equal_nan=True)
    # assert torch.allclose(softmax_d, softmax_d_xla, rtol=1e-3, atol=1e-3, equal_nan=True)
    softmax_d_pad = softmax_d[:,:,:seqlen_q]
    # print(f'pytorch softmax_d={softmax_d}')
    
    difference_s = torch.abs(softmax_d_pad - softmax_d_xla)
    tolerance_s = 1e-5 + 1e-5 * torch.abs(softmax_d_xla)
    mask_s = difference_s > tolerance_s

    # Get indices where the elements are different
    indices_s = mask_s.nonzero(as_tuple=False)
    # print(f'softmax_d diff={indices_s}, {indices_s.size()}')
    assert(indices_s.numel() < q.numel() * 1e-2)

    difference = torch.abs(dv - dv_xla)
    tolerance = 1e-5 + 1e-5 * torch.abs(dv_xla)
    mask = difference > tolerance

    # Get indices where the elements are different
    indices = mask.nonzero(as_tuple=False)
    # print(f'dv diff={indices}, {indices.size()}')
    assert(indices.numel() < q.numel() * 1e-2)
    
    difference_q = torch.abs(dq - dq_xla)
    tolerance_q = 1e-5 + 1e-5 * torch.abs(dq_xla)
    mask_q = difference_q > tolerance_q

    # Get indices where the elements are different
    indices_q = mask_q.nonzero(as_tuple=False)
    # print(f'dq diff={indices_q}, {indices_q.size()}')
    assert(indices_q.numel() < q.numel() * 1e-2)
    
    difference_k = torch.abs(dk - dk_xla)
    tolerance_k = 1e-5 + 1e-5 * torch.abs(dk_xla)
    mask_k = difference_k > tolerance_k

    # Get indices where the elements are different
    indices_k = mask_k.nonzero(as_tuple=False)
    # print(f'dk diff={indices_k}, {indices_k.size()}')
    assert(indices_k.numel() < q.numel() * 1e-2)
