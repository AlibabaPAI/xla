from types import MethodType

import torch
from torch.distributed.utils import _apply_to_tensors
from torch.utils.checkpoint import check_backward_validity, detach_variable
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.utils.checkpoint import checkpoint


def checkpoint_module(module):
  """
  Wrap a `module`'s `forward` method with gradient checkpointing (also called
  activation checkpointing) via `torch_xla.utils.checkpoint.checkpoint`.
  """

  def _xla_checkpointed_forward_no_kwargs(m, num_args, num_kwargs,
                                          *packed_args):
    # unpack packed_args into args and kwargs
    assert num_args + num_kwargs * 2 == len(packed_args)
    args = packed_args[:num_args]
    kwargs = packed_args[num_args:]
    kwargs = dict(zip(kwargs[:num_kwargs], kwargs[num_kwargs:]))
    return m._xla_checkpointed_forward_original(*args, **kwargs)

  def _forward_with_checkpoint(m, *args, **kwargs):
    # pack args and kwargs together as `torch_xla.utils.checkpoint.checkpoint`
    # doesn't support keyword arguments
    packed_args = args + tuple(kwargs.keys()) + tuple(kwargs.values())
    input_requires_grad = any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in packed_args)
    if input_requires_grad:
      outputs = checkpoint(m._xla_checkpointed_forward_no_kwargs, len(args),
                           len(kwargs), *packed_args)
    else:
      # No input requires gradients so we won't checkpoint this forward pass.
      # Note that `m`` might have parameters that require gradients, but they
      # are beyond what `torch_xla.utils.checkpoint.checkpoint` can handle.
      outputs = m._xla_checkpointed_forward_original(*args, **kwargs)
    return outputs

  assert isinstance(module, torch.nn.Module)
  # replace `module`'s forward method with its checkpointed version
  module._xla_checkpointed_forward_original = module.forward
  module._xla_checkpointed_forward_no_kwargs = MethodType(
      _xla_checkpointed_forward_no_kwargs, module)
  module.forward = MethodType(_forward_with_checkpoint, module)
  return module


def dummy_all_gather(value, dim=0, groups=None):
  """A dummy op for debugging with the same output shape as all_gather"""
  repeat_num = [1] * value.dim()
  repeat_num[dim] = xm.xrt_world_size()
  return value.repeat(tuple(repeat_num))


def dummy_all_reduce(reduce_type, inputs, scale=1.0, groups=None):
  """A dummy op for debugging with the same output shape as all_reduce"""
  if isinstance(inputs, torch.Tensor):
    return inputs * scale
  return [t.mul_(scale) for t in inputs]


def dummy_reduce_scatter(reduce_type,
                         input,
                         scale,
                         scatter_dim,
                         shard_count,
                         groups=None):
  """A dummy op for debugging with the same output shape as reduce_scatter"""
  assert shard_count == xm.xrt_world_size()
  full_size = input.size(scatter_dim)
  shard_size = full_size // xm.xrt_world_size()
  begin = shard_size * xm.get_ordinal()
  end = begin + shard_size
  slices = [None] * input.dim()
  slices[scatter_dim] = slice(begin, end)
  return input[tuple(slices)] * scale


class XLAPatchedLinear(torch.autograd.Function):
  """
  A patched version of `torch.nn.functional.linear` with explicitly-defined backward
  as a workaround to https://github.com/pytorch/xla/issues/3811.

  Modified from https://pytorch.org/docs/stable/notes/extending.html#example
  """

  @staticmethod
  def forward(ctx, input, weight, bias=None):
    # bias is an optional argument
    ctx.save_for_backward(input, weight, bias)
    with torch.no_grad():
      return torch._C._nn.linear(input, weight, bias)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    input = input.to(grad_output)

    input_dim = input.dim()
    if input_dim > 2:
      input_flat = input.flatten(start_dim=0, end_dim=-2)
      grad_output_flat = grad_output.flatten(start_dim=0, end_dim=-2)
    else:
      input_flat = input
      grad_output_flat = grad_output

    if ctx.needs_input_grad[0]:
      grad_input_flat = grad_output_flat.mm(weight)
      if input_dim > 2:
        grad_input = grad_input_flat.view(*input.size())
      else:
        grad_input = grad_input_flat
    if ctx.needs_input_grad[1]:
      grad_weight = grad_output_flat.t().mm(input_flat)
    if bias is not None and ctx.needs_input_grad[2]:
      grad_bias = grad_output_flat.sum(0)

    return grad_input, grad_weight, grad_bias


def _xla_patched_nn_linear_forward(m, input):
  return XLAPatchedLinear.apply(input, m.weight, m.bias)


def apply_xla_patch_to_nn_linear(module,
                                 patched_function=_xla_patched_nn_linear_forward
                                ):
  """
  Recursively apply a patch to the forward pass of `nn.Linear` layers
  to enable using `XLAPatchedLinear.apply` as `torch.nn.functional.linear`,
  so that the backward pass will explicitly use the weight parameter of an
  `nn.Linear` layer to resolve https://github.com/pytorch/xla/issues/3811.

  Without this patch, an `nn.Linear` module in PyTorch/XLA holds and uses
  an intermediate result (rather than the weight parameter) in its backward
  computation, which may break the FSDP's full parameter freeing on it.
  """

  def _try_patching_forward_method(m, forward_method_name="forward"):
    # Check if the module's forward signature is same as in `nn.Linear`
    # (if it has already been modified through other means, we will skip the
    # patch to its forward method here).
    forward_method = getattr(m, forward_method_name, None)
    if forward_method is None:
      return
    if getattr(forward_method, "__func__", None) != torch.nn.Linear.forward:
      return

    patched_forward_method = MethodType(patched_function, m)
    m._nn_linear_forward_original = forward_method
    setattr(m, forward_method_name, patched_forward_method)

  for m in module.modules():  # includes self
    if isinstance(m, torch.nn.Linear):
      _try_patching_forward_method(m, "forward")
      # also handle the case of gradient checkpointing via `checkpoint_module`
      _try_patching_forward_method(m, "_xla_checkpointed_forward_original")

  return module


class AutogradFunction(torch.autograd.Function):
  """
  This class creates a separate computational graph during the forward pass and
  calls `torch.autograd.backward` during the backward pass. This class allows the
  reduce_scatter hook of FSDP to be correctly triggered during backward when flatten
  parameters is enabled, rather than triggering the reduce_scatter hooks collectively
  at the end of the backward pass.
  """

  @staticmethod
  def forward(ctx, run_function, *args):
    check_backward_validity(args)
    args = detach_variable(tuple(args))

    # Save non-tensor inputs in ctx, keep a placeholder None for tensors
    # to be filled out during the backward.
    ctx.inputs = []
    ctx.input_tensor_indices = []
    tensor_inputs = []

    for i, arg in enumerate(args):
      if torch.is_tensor(arg):
        tensor_inputs.append(arg)
        ctx.input_tensor_indices.append(i)
        ctx.inputs.append(None)
      else:
        ctx.inputs.append(arg)

    with torch.enable_grad():
      outputs = run_function(*args)

    ctx.outputs = []
    ctx.output_tensor_indices = []
    tensor_outputs = []
    if torch.is_tensor(outputs):
      tensor_outputs = [outputs]
      ctx.output_tensor_indices = [0]
      ctx.outputs = [None]
    else:
      assert isinstance(outputs,
                        (list, tuple)), "Only support for tuple or list output"
      for i, out in enumerate(outputs):
        if torch.is_tensor(out):
          tensor_outputs.append(out)
          ctx.output_tensor_indices.append(i)
          ctx.outputs.append(None)
        else:
          ctx.outputs.append(out)

    ctx.save_for_backward(*(tensor_inputs + tensor_outputs))
    outputs = _apply_to_tensors(lambda t: t.clone().detach(), outputs)

    return outputs

  @staticmethod
  def backward(ctx, *args):
    if not torch.autograd._is_checkpoint_valid():
      raise RuntimeError(
          "AutogradFunction is not compatible with .grad() or when an `inputs` parameter"
          " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
          " argument.")
    tensors = ctx.saved_tensors

    # Copy the list to avoid modifying original list.
    inputs = list(ctx.inputs)
    input_tensor_indices = ctx.input_tensor_indices
    # Fill in inputs with appropriate saved tensors.
    for i, idx in enumerate(input_tensor_indices):
      inputs[idx] = tensors[i]

    outputs = list(ctx.outputs)
    output_tensor_indices = ctx.output_tensor_indices
    # Fill in outputs with appropriate saved tensors.
    for i, idx in enumerate(output_tensor_indices):
      outputs[idx] = tensors[len(input_tensor_indices) + i]

    # run backward() with only tensor that requires grad
    outputs_with_grad = []
    args_with_grad = []
    for i in range(len(outputs)):
      if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
        outputs_with_grad.append(outputs[i])
        args_with_grad.append(args[i])
    if len(outputs_with_grad) == 0:
      raise RuntimeError("none of output has requires_grad=True,"
                         " this autograd_module() is not necessary")
    torch.autograd.backward(outputs_with_grad, args_with_grad)
    grads = tuple(
        inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)

    return (None,) + grads


def autograd_module(module):
  """
  Wrap a `module`'s `forward` method with `AutogradFunction`.
  """

  def _xla_autograd_forward_no_kwargs(m, num_args, num_kwargs, *packed_args):
    # unpack packed_args into args and kwargs
    assert num_args + num_kwargs * 2 == len(packed_args)
    args = packed_args[:num_args]
    kwargs = packed_args[num_args:]
    kwargs = dict(zip(kwargs[:num_kwargs], kwargs[num_kwargs:]))
    return m._xla_autograd_forward_original(*args, **kwargs)

  def _forward_with_autograd(m, *args, **kwargs):
    # pack args and kwargs together as `AutogradFunction`
    # doesn't support keyword arguments
    packed_args = args + tuple(kwargs.keys()) + tuple(kwargs.values())
    input_requires_grad = any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in packed_args)
    if input_requires_grad:
      outputs = AutogradFunction.apply(m._xla_autograd_forward_no_kwargs,
                                       len(args), len(kwargs), *packed_args)
    else:
      # No input requires gradients so we won't wrap this forward pass.
      # Note that `m`` might have parameters that require gradients, but they
      # are beyond what `AutogradFunction` can handle.
      outputs = m._xla_autograd_forward_original(*args, **kwargs)
    return outputs

  assert isinstance(module, torch.nn.Module)
  # replace `module`'s forward method with its autograd version
  module._xla_autograd_forward_original = module.forward
  module._xla_autograd_forward_no_kwargs = MethodType(
      _xla_autograd_forward_no_kwargs, module)
  module.forward = MethodType(_forward_with_autograd, module)
  return module


def get_tensor_id(t):
  return torch_xla._XLAC._xla_get_tensor_id(t)
