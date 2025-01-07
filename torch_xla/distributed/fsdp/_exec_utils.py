class ExecState:

  def __init__(self):
    self._index_to_fsdp_module = {}
    self._fsdp_module_to_index = {}
    self._iter = 0
    self._current_index = 0

  @property
  def is_first_iter(self) -> bool:
    return self._iter == 0

  def record_forward(self, fsdp_module):
    if self.is_first_iter:
      assert fsdp_module not in self._fsdp_module_to_index
      self._index_to_fsdp_module[self._current_index] = fsdp_module
      self._fsdp_module_to_index[fsdp_module] = self._current_index
      self._current_index += 1
    else:
      assert fsdp_module in self._fsdp_module_to_index
      assert self._fsdp_module_to_index[fsdp_module] == self._current_index, \
          "FSDP module is not in the same execution order as first iteration."
      self._current_index += 1

  def get_prefetch_module(self):
    if self.is_first_iter:
      return None
    if self._current_index >= len(self._index_to_fsdp_module):
      return None
    return self._index_to_fsdp_module[self._current_index]

  def next_iter(self):
    self._iter += 1
    self._current_index = 0
