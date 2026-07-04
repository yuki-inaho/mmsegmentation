from mmseg.registry import OPTIMIZERS
from ranger_adabelief import RangerAdaBelief


@OPTIMIZERS.register_module()
class AdaBelief(RangerAdaBelief):
    # NOTE (cu12): the first constructor parameter must be named ``params``.
    # mmengine's DefaultOptimWrapperConstructor inspects the optimizer class's
    # first parameter name and fills it with the model parameters
    # (mmengine/optim/optimizer/default_constructor.py). A bare ``*args``
    # signature makes that first name ``args``, so mmengine would pass
    # ``args=<params>`` and RangerAdaBelief raises an unexpected-keyword error.
    def __init__(self, params, *args, **kwargs):
        # SGD-style keys that may leak in from shared optimizer configs.
        kwargs.pop("momentum", None)
        kwargs.pop("nesterov", None)
        super(AdaBelief, self).__init__(params, *args, **kwargs)
