import torch_xla

# Whether to skip checking input is a device data or not in the optim_mod.
# Enabling it will reduce the overhead a bit but will throw a runtime error
# if input is a pending IR.
skip_input_data_check = False

# Whether to transform the FX graph into an XLA computation
# and creating a call node for that computation. This allows XLA to trace
# a more extensive computation graph, potentially leading to greater
# optimization opportunities.
use_call_computation = False

# The model outside dynamo is on cuda or not.
outside_on_cuda = False

# Whether to mark step after each layer when early_sync happens.
mark_step_after_layer_if_early_sync = False

# whether to remove the sync of xla run_cached_graph in dynamo + xla backend.
no_xla_graph_sync = False