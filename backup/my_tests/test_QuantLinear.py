from quantize.blocks.int_linear import QuantLinear
import torch
from quantize.blocks.block_wise_quant_config_search import kldiv, mse
l = torch.nn.Linear(100, 100)


weight_quant_params = {
    # "n_bits": args.wbits,
    "per_channel_axes": [0],
    "symmetric": False,
    "dynamic_method": "per_channel",
    # "group_size": args.group_size,
    # "lwc":args.lwc,
    "disable_zero_point": False,
}
act_quant_params = {
    # "n_bits":  args.abits,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": "per_token",
}

q = QuantLinear(l, weight_quant_params, act_quant_params, wbits=4, abits=4)
q.set_quant_state(weight_quant=True, act_quant=True)

inputs = torch.randn(10, 100)
out_l = l.forward(inputs)
out_q = q.forward(inputs)

print(kldiv(out_q, out_l))
print(mse(out_q, out_l))


# python -m my_tests.test_QuantLinear
