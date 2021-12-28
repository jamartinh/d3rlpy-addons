import torch


class WrapDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        if name in dir(self):
            return super().__getattr__(name)
        else:
            return getattr(super().__getattr__("module"), name)


def parallelize_algo(algo):
    """
    Simply call parallelize_algo(algo) after build with env or dataset
     to parallelize on all GPUs available.
    Args:
        algo: a d3rlpy.algo

    Returns: None

    """
    black_list = [
        "policy",
        "q_function",
        "policy_optim",
        "q_function_optim",
    ]  # special properties
    keys = [k for k in dir(algo.impl) if k not in black_list]
    for key in keys:
        module = getattr(algo.impl, key)
        if isinstance(module, torch.nn.Module) and not isinstance(module, torch.nn.DataParallel):
            data_parallel_module = WrapDataParallel(module)
            setattr(algo.impl, key, data_parallel_module)


