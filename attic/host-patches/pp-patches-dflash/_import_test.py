import torch
from torch import nn
import vllm.models.glm5next.nvidia.model as m
import vllm.v1.worker.gpu.model_runner as mr
import vllm.v1.worker.gpu.spec_decode.dflash.utils as du
from vllm.model_executor.models.interfaces import supports_eagle3, EagleModelMixin
print('imports ok')
for cls in (m.Glm5NextForCausalLM, m.Glm5NextForConditionalGeneration):
    print(cls.__name__, 'supports_eagle3(class)=', supports_eagle3(cls), 'flag=', cls.supports_pp_aux_hidden_states)
from types import SimpleNamespace
mod = m.Glm5NextModel.__new__(m.Glm5NextModel); nn.Module.__init__(mod)
mod.config = SimpleNamespace(num_hidden_layers=45)
mod._set_aux_hidden_state_layers((43, 6, 25, 15, 34))
print('sorted ids', mod.aux_hidden_state_layers)
for start in (0, 23):
    mod.start_layer = start
    print('start_layer', start, 'incoming', mod.incoming_aux_ids(), [mod.aux_key(i) for i in mod.incoming_aux_ids()])
try:
    mod._set_aux_hidden_state_layers((0, 46))
except ValueError as e: print('range check:', e)
from vllm.model_executor.layers.mhc import hc_contract
x = torch.arange(24., dtype=torch.bfloat16).reshape(2, 4, 3)
print('hc_contract', hc_contract(x, 4).tolist(), 'dtype', hc_contract(x, 4).dtype)
from vllm.distributed.utils import get_pp_indices
print('pp indices', [get_pp_indices(45, r, 2) for r in range(2)])
