"""Minimal BasicSR initialization helper required by the GFPGAN architecture.

Derived from XPixelGroup/BasicSR 1.4.2 under Apache-2.0. This isolated helper
replaces the BasicSR runtime dependency; no distributed-launch or subprocess
code is included.
"""

from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize convolution, linear, and batch-normalization modules."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for child in module.modules():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(child.weight, **kwargs)
                child.weight.data *= scale
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)
            elif isinstance(child, _BatchNorm):
                init.constant_(child.weight, 1)
                if child.bias is not None:
                    child.bias.data.fill_(bias_fill)
